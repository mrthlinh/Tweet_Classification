import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.clustering.DistributedLDAModel
//import spark.implicits._
object TweetTopicModel {
  def main(args: Array[String]){
    // Initialize SparkSession
    val spark = SparkSession.builder()
      .appName("Twitter Comment Topic Model")
      .getOrCreate()

    val input_path = args(0)
    val output_path = args(1)

//    val input_path = "D:\\Tweets.csv"
//    val output_path = "D:\\OUTPUT"

    import spark.implicits._ // << add this
    // 1. Load the dataset
    val originalCorpus  = spark.read.option("header","true").csv(input_path)
    val filterCorpus = originalCorpus.filter("text is not null").select("airline_sentiment","airline","text").withColumn("text",regexp_replace($"text","@",""))
    // val filterCorpus1 = filterCorpus.withColumn("text",regexp_replace($"text","@",""))
    println("length of data: "+originalCorpus.count)
    println("length of filtered data: "+filterCorpus.count)

    // 2. Find the best and worst airline
    val myData = filterCorpus.select("airline_sentiment","airline")
      .withColumn("airline_sentiment_num",when(filterCorpus("airline_sentiment") === "neutral", 2.5)
        .when(filterCorpus("airline_sentiment") === "positive", 5.0).otherwise(1.0))

    // Group by Airline Brand -> calculate averate rating for each airline -> rename and sort desc
    val aveRating = myData.groupBy("airline").agg(avg("airline_sentiment_num").alias("ave_rating")).sort($"ave_rating".desc)
//    aveRating.show

    val best = aveRating.select("airline").take(1)
    val bestAirline = best(0).getString(0)
    val worst = aveRating.sort($"ave_rating".asc).select("airline").take(1)
    val worstAirline = worst(0).getString(0)

    val twoAirline = filterCorpus.filter(col("airline") === bestAirline || col("airline") === worstAirline )

    // remove stop words
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val stopWord = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("filtered")
    // Set params for CountVectorizer
    val vectorizer = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("features")
      .setVocabSize(2048)
    val lda = new LDA()
      .setK(2)
      .setMaxIter(100)
      .setOptimizer("em")
    // Pipeline
    val pipeline = new Pipeline().setStages(Array(tokenizer,stopWord, vectorizer,lda))

    val pipelineModel = pipeline.fit(twoAirline)
    val trainData = pipelineModel.transform(twoAirline).select("features")

    val vectorizerModel = pipelineModel.stages(2).asInstanceOf[CountVectorizerModel]
    val ldaModel = pipelineModel.stages(3).asInstanceOf[DistributedLDAModel]

    val vocabList = vectorizerModel.vocabulary
    val termsIdx2Str = udf { (termIndices: Seq[Int]) => termIndices.map(idx => vocabList(idx)) }

    val topics = ldaModel.describeTopics(maxTermsPerTopic = 20)
      .withColumn("terms", termsIdx2Str(col("termIndices")))

//    val rdd_result = topics.rdd.coalesce(1)
    val rdd_result = topics.rdd
    rdd_result.saveAsTextFile(output_path)

  }
}
