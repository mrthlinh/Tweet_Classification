import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.sql.functions._

object FrequentItemAnalysis {
  def main(args: Array[String]): Unit ={
    // Initialize SparkSession
//    val spark = SparkSession.builder()
//      .appName("FPG")
//      .getOrCreate()
//
    val input_path = args(0)
    val output_path1 = args(1)
    val output_path2 = args(2)
//    val input_path = "D:\\order_products__train.csv"
//    val output_path1 = "D:\\output_FPG1"
//    val output_path2 = "D:\\output_FPG2"
//    val sc = new SparkContext(new SparkConf().setAppName("Spark Count").setMaster("local"))

    val sc = new SparkContext(new SparkConf().setAppName("Spark Count"))
//    val sc = spark.sparkContext
    val pro = sc.textFile(input_path)
    val header = pro.first
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    val data = pro.filter(line => line != header).map(s => (s.trim.split(',')(0),s.trim.split(',')(1))).toDF()
    val items = data.groupBy("_1").agg(collect_list("_2") as "item").orderBy($"_1").select($"item")
    val fpgrowth = new FPGrowth().setItemsCol("item").setMinSupport(0.011).setMinConfidence(0.0)
    val model = fpgrowth.fit(items)

    val result_freq = model.freqItemsets.sort($"freq".desc)
    val rdd_result1 = result_freq.limit(10).rdd
    rdd_result1.saveAsTextFile(output_path1)

    val result_assoc = model.associationRules.sort($"confidence".desc)
    val rdd_result2 = result_assoc.limit(10).rdd
    rdd_result2.saveAsTextFile(output_path2)
//    rdd_result.saveAsTextFile(output_path)

//    result.limit(10).write.format("csv").option("header","true").csv(output_path)
  }
}
