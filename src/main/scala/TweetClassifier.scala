import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.sql.SparkSession

object TweetClassifier  {
  def main(args: Array[String]) {

    // Initialize SparkSession
    val spark = SparkSession.builder()
      .appName("Twitter Comment Sentiment")
      .getOrCreate()

    val input_path = args(0)
    val output_path = args(1)

//    val input_path = "D:\\Tweets.csv"
//    val output_path = "D:\\OUTPUT"

    // 1. Loading: First step is to define an input argument that defines the path from which to load the dataset. After that, you will need to remove rows where the text field is null.
    val data = spark.read.option("header", "true").csv(input_path)

    val data_filter = data.filter("text is not null")

    // 2. Pre-Processing: You will start by creating a pre-processing pipeline with the following stages:
    // Configure an ML pipeline, which consists of three stages: tokenizer, Stop Word Remover, hashingTF, and lr.
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val stopWord = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("stopWordFilter")
    val hashingTF = new HashingTF().setInputCol(stopWord.getOutputCol).setOutputCol("features")
    val labelCon = new StringIndexer().setInputCol("airline_sentiment").setOutputCol("label")
    val pipeline = new Pipeline().setStages(Array(tokenizer, stopWord,hashingTF,labelCon))

    val processor = pipeline.fit(data_filter)
    val processData = processor.transform(data_filter).select("tweet_id","features","label")

    // Prepare test data
    val splits = processData.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val test = splits(1).toDF()

    // 3. Model Creation - You will need to create two classification models that you can select from the MLlib classification library
    // A. LogisticRegression one-vs-all
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setTol(1E-6)
      .setFitIntercept(true)

    val ovr_lr = new OneVsRest().setClassifier(lr)

    val paramGrid_lr = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.001, 0.1))
      .build()

    val cv_lr = new CrossValidator()
      .setEstimator(ovr_lr)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid_lr)
      .setNumFolds(2)

    val cvModel_lr = cv_lr.fit(processData)

    // B. SVM one-vs-all
    // SVM
    val lsvc = new LinearSVC()
      .setMaxIter(10)

    val ovr_lsvc = new OneVsRest().setClassifier(lsvc)

    val paramGrid_lsvc = new ParamGridBuilder()
      .addGrid(lsvc.regParam, Array(0.001, 0.1))
      .build()

    val cv_lsvc = new CrossValidator()
      .setEstimator(ovr_lsvc)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid_lsvc)
      .setNumFolds(2)  // Use 3+ in practice

    // Run cross-validation, and choose the best set of parameters.
    val cvModel_lsvc = cv_lsvc.fit(processData)


//    Get Model evaluation of 2 models

    val result_lsvc = ModelEvaluation(cvModel_lsvc,test)
    val result_lr = ModelEvaluation(cvModel_lr,test) // try to define a function

//    Export result
    val result = Array(("SVM", result_lsvc), ("LR", result_lr))
    val rdd_result = spark.sparkContext.parallelize(result).coalesce(1)
    rdd_result.saveAsTextFile(output_path)
  }

  def ModelEvaluation(cvmodel: org.apache.spark.ml.tuning.CrossValidatorModel, test: org.apache.spark.sql.DataFrame): Double = {
    val rdd_data = cvmodel.transform(test).select("prediction","label").rdd
    val predictionAndLabels = rdd_data.map{case(x) =>(x.getDouble(0),x.getDouble(1))}

    // Instantiate metrics object
    val metrics = new MulticlassMetrics(predictionAndLabels)

    // Overall Statistics
    val accuracy = metrics.accuracy
    return accuracy
  }


}
