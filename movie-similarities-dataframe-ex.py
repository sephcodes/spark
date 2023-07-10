# we're gonna select userID,movieID,rating
# find every movie pair rated by the same user
#   done with a self-join operation (very intensive operation)
#   that gives us movie1,movie2,rating1,rating2
#   filter out dupe pairs (movie1,movie2) = (movie2,movie1)
# compute cosine similarity scores for every pair
#   add x^2, y^2, x*y columns
#   group by (movie1,movie2) pairs
#   compute similarity score for each aggregated pair
# filter, sort, and display results

from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, FloatType
import sys
# load up u.item file locally from driver script before broadcasting it out
import codecs

def loadMovieNames():
    movieNames = {}
    with codecs.open("ml-100k/u.ITEM", "r", encoding='ISO-8859-1', errors='ignore') as f:
        for line in f:
            line = line.replace('\n','')
            fields = line.split('|')
            movieNames[int(fields[0])] = fields
    return movieNames

spark = SparkSession.builder.appName("PopularMovies").getOrCreate()
nameDict = spark.sparkContext.broadcast(loadMovieNames())

def computeCosineSimilarity(spark, data):
    # Compute xx, xy and yy columns as new columns
    pairScores = data \
      .withColumn("xx", func.col("rating1") * func.col("rating1")) \
      .withColumn("yy", func.col("rating2") * func.col("rating2")) \
      .withColumn("xy", func.col("rating1") * func.col("rating2")) 

    # Compute numerator, denominator and numPairs columns
    # group together all user ratings for that distinct pair of movies
    calculateSimilarity = pairScores \
      .groupBy("movie1", "movie2") \
      .agg( \
        func.sum(func.col("xy")).alias("numerator"), \
        (func.sqrt(func.sum(func.col("xx"))) * func.sqrt(func.sum(func.col("yy")))).alias("denominator"), \
        func.count(func.col("xy")).alias("numPairs")
      )

    # Calculate score and select only needed columns (movie1, movie2, score, numPairs)
    result = calculateSimilarity \
      .withColumn("score", \
        func.when(func.col("denominator") != 0, func.col("numerator") / func.col("denominator")) \
          .otherwise(0) \
      ).select("movie1", "movie2", "score", "numPairs")

    return result

# Get movie name by given movie id 
def getMovieName(movieNames, movieId):
    result = movieNames.filter(func.col("movieID") == movieId) \
        .select("movieTitle").collect()[0]

    return result[0]

def jaccard(movieId1, movieId2) -> float:
    movie1genres = list(map(int, nameDict.value[movieId1][5:]))
    movie2genres = list(map(int, nameDict.value[movieId2][5:]))
    
    unn = 0
    intr = 0
    for i in range(19):
        sum = movie1genres[i] + movie2genres[i]
        if(sum >= 1):
            unn += 1
            if(sum == 2):
                intr += 1
    
    return float(intr)/unn

jaccardUDF = func.udf(jaccard)

# .master("local[*]") means want to use every cpu core on system to execute job
# this is very dangerous when deployed on clusters
# spark = SparkSession.builder.appName("MovieSimilarities").master("local[*]").getOrCreate()

movieNamesSchema = StructType([ \
                               StructField("movieID", IntegerType(), True), \
                               StructField("movieTitle", StringType(), True), \
                               StructField("releaseDate", IntegerType(), True), \
                               StructField("videoDate", IntegerType(), True), \
                               StructField("imdbUrl", IntegerType(), True), \
                               StructField("unknown", IntegerType(), True), \
                               StructField("action", IntegerType(), True), \
                               StructField("adventure", IntegerType(), True), \
                               StructField("animation", IntegerType(), True), \
                               StructField("children", IntegerType(), True), \
                               StructField("comedy", IntegerType(), True), \
                               StructField("crime", IntegerType(), True), \
                               StructField("documentary", IntegerType(), True), \
                               StructField("drama", IntegerType(), True), \
                               StructField("fantasy", IntegerType(), True), \
                               StructField("filmNoir", IntegerType(), True), \
                               StructField("horror", IntegerType(), True), \
                               StructField("musical", IntegerType(), True), \
                               StructField("mystery", IntegerType(), True), \
                               StructField("romance", IntegerType(), True), \
                               StructField("scifi", IntegerType(), True), \
                               StructField("thriller", IntegerType(), True), \
                               StructField("war", IntegerType(), True), \
                               StructField("western", IntegerType(), True)])
    
moviesSchema = StructType([ \
                     StructField("userID", IntegerType(), True), \
                     StructField("movieID", IntegerType(), True), \
                     StructField("rating", IntegerType(), True), \
                     StructField("timestamp", LongType(), True)])
    
    
# Create a broadcast dataset of movieID and movieTitle.
# Apply ISO-885901 charset
movieNames = spark.read \
      .option("sep", "|") \
      .option("charset", "ISO-8859-1") \
      .schema(movieNamesSchema) \
      .csv("ml-100k/u.item")

# Load up movie data as dataset
movies = spark.read \
      .option("sep", "\t") \
      .schema(moviesSchema) \
      .csv("ml-100k/u.data")


ratings = movies.select("userId", "movieId", "rating")

# Emit every movie rated together by the same user.
# Self-join to find every combination.
# Select movie pairs and rating pairs
# ratings becomes aliased to ratings 1 and then give it another alias of ratings2 and join together
# self join gives unique movieID pairs (each unique pair for every user)
# will join on userIDs are equal
# & condition removes dupes
# .select formats new DF
moviePairs = ratings.alias("ratings1") \
      .join(ratings.alias("ratings2"), (func.col("ratings1.userId") == func.col("ratings2.userId")) \
            & (func.col("ratings1.movieId") < func.col("ratings2.movieId"))) \
      .select(func.col("ratings1.movieId").alias("movie1"), \
        func.col("ratings2.movieId").alias("movie2"), \
        func.col("ratings1.rating").alias("rating1"), \
        func.col("ratings2.rating").alias("rating2"))


moviePairSimilarities = computeCosineSimilarity(spark, moviePairs).cache()

jaccardDF = moviePairSimilarities.withColumn("jaccard", jaccardUDF(func.col("movie1"), func.col("movie2")))

if (len(sys.argv) > 1):
    # enforce quality threshold
    scoreThreshold = 0.90
    # at least 50 users rated the pair
    coOccurrenceThreshold = 50.0
    # use high jaccard
    jaccardThreshold = 0.5

    # extract movie we want from command line (as a CL parameter)
    # spark-submit file_name.py 50
    movieID = int(sys.argv[1])

    # Filter for movies with this sim that are "good" as defined by
    # our quality thresholds above
    filteredResults = jaccardDF.filter( \
        ((func.col("movie1") == movieID) | (func.col("movie2") == movieID)) & \
          (func.col("score") > scoreThreshold) & (func.col("numPairs") > coOccurrenceThreshold) & \
          (func.col("jaccard") > jaccardThreshold))

    print ("Top 10 similar movies for " + getMovieName(movieNames, movieID))
    
    # our movieID is sometimes movie1 or movie2, so we want not it
    def chooseCorrectMovieID(movie1, movie2, movieID):
        if(movie1 == movieID):
            return movie2
        else:
            return movie1
        
    chooseCorrectMovieIDUDF = func.udf(chooseCorrectMovieID)

    resultsDF = filteredResults.withColumn("movieID", chooseCorrectMovieIDUDF(func.col("movie1"), func.col("movie2"), func.lit(movieID)))
    resultsDF = resultsDF.select(func.col("movieID"), \
        func.col("score"), func.col("jaccard"))
    resultsDF = resultsDF.join(movieNames, "movieID").select(func.col("movieTitle"), \
        func.col("score"), func.col("jaccard")).sort(func.col("score").desc())
    resultsDF.show()
