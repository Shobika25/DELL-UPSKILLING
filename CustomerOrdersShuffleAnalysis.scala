
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

case class CustomerOrder(
  order_id: Long,
  customer_id: String,
  customer_name: String,
  product: String,
  category: String,
  quantity: Int,
  order_amount: Long,
  status: String
)

object CustomerOrdersShuffleAnalysis {

  def main(args: Array[String]): Unit = {

    // ==========================================
    // 1. SPARK CONFIGURATION
    // ==========================================

    val conf = new SparkConf()
      .setAppName("CustomerOrdersShuffleAnalysis")
      .setMaster("local[*]")
      .set("spark.serializer",
        "org.apache.spark.serializer.JavaSerializer")
      .set("spark.sql.shuffle.partitions", "4")

    val spark = SparkSession.builder()
      .config(conf)
      .config("spark.serializer",
        "org.apache.spark.serializer.JavaSerializer")
      .getOrCreate()

    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    println("ACTIVE SERIALIZER: " +
      sc.getConf.get("spark.serializer", "DEFAULT"))

    import spark.implicits._

    // ==========================================
    // 2. CREATE SAMPLE DATA
    // ==========================================

    val orders = Seq(
      CustomerOrder(1001, "C101", "Anita", "Laptop", "Electronics", 1, 55000, "Delivered"),
      CustomerOrder(1002, "C102", "Bharat", "Phone", "Electronics", 2, 30000, "Delivered"),
      CustomerOrder(1003, "C101", "Anita", "Mouse", "Accessories", 2, 1000, "Delivered"),
      CustomerOrder(1004, "C103", "Charan", "Chair", "Furniture", 1, 7000, "Cancelled"),
      CustomerOrder(1005, "C104", "Divya", "Desk", "Furniture", 1, 12000, "Delivered"),
      CustomerOrder(1006, "C102", "Bharat", "Headphones", "Accessories", 1, 2500, "Returned"),
      CustomerOrder(1007, "C105", "Esha", "Phone", "Electronics", 1, 15000, "Delivered"),
      CustomerOrder(1008, "C103", "Charan", "Desk", "Furniture", 2, 24000, "Delivered"),
      CustomerOrder(1009, "C101", "Anita", "Keyboard", "Accessories", 1, 2000, "Returned"),
      CustomerOrder(1010, "C104", "Divya", "Laptop", "Electronics", 1, 55000, "Delivered"),
      CustomerOrder(1011, "C102", "Bharat", "Mouse", "Accessories", 3, 1500, "Delivered"),
      CustomerOrder(1012, "C105", "Esha", "Chair", "Furniture", 2, 14000, "Cancelled")
    )

    val orderRDD: RDD[CustomerOrder] =
      sc.parallelize(orders, 4)

    val orderDF: DataFrame =
      orders.toDF()

    println("\n========== CUSTOMER ORDERS ==========")
    orderDF.show(false)

    // ==========================================
    // 3. PROBLEM 1: CUSTOMER-WISE TOTAL VALUE
    // ==========================================

    println("\n========== PROBLEM 1: RDD ==========")

    val customerRDD = orderRDD
      .map(order =>
        (order.customer_id,
          (order.customer_name, order.order_amount))
      )
      .reduceByKey {
        case ((name1, amount1), (_, amount2)) =>
          (name1, amount1 + amount2)
      }
      .map {
        case (customerId, (customerName, total)) =>
          (customerId, customerName, total)
      }
      .sortBy(
        row => (-row._3, row._1),
        ascending = true,
        numPartitions = sc.defaultParallelism
      )

    val customerRDDResult = customerRDD.collect().toSeq

    customerRDDResult.foreach(println)

    println("\n========== PROBLEM 1: DATAFRAME ==========")

    val customerDF = orderDF
      .groupBy("customer_id", "customer_name")
      .agg(
        sum("order_amount").as("total_order_value")
      )
      .orderBy(
        col("total_order_value").desc,
        col("customer_id").asc
      )

    customerDF.show(false)

    val customerDFResult = customerDF.collect().map { row =>
      (
        row.getAs[String]("customer_id"),
        row.getAs[String]("customer_name"),
        row.getAs[Long]("total_order_value")
      )
    }.toSeq

    assert(
      customerRDDResult == customerDFResult,
      "Problem 1 results do not match!"
    )

    println("Problem 1: RDD and DataFrame results MATCH")

    // ==========================================
    // 4. PROBLEM 2: PRODUCT-WISE UNITS SOLD
    // ==========================================

    println("\n========== PROBLEM 2: RDD ==========")

    val productRDD = orderRDD
      .map(order =>
        (order.product, order.quantity.toLong)
      )
      .reduceByKey(_ + _)
      .sortBy(
        row => (-row._2, row._1),
        ascending = true,
        numPartitions = sc.defaultParallelism
      )

    val productRDDResult = productRDD.collect().toSeq

    productRDDResult.foreach(println)

    println("\n========== PROBLEM 2: DATAFRAME ==========")

    val productDF = orderDF
      .groupBy("product")
      .agg(
        sum("quantity").as("units_sold")
      )
      .orderBy(
        col("units_sold").desc,
        col("product").asc
      )

    productDF.show(false)

    val productDFResult = productDF.collect().map { row =>
      (
        row.getAs[String]("product"),
        row.getAs[Long]("units_sold")
      )
    }.toSeq

    assert(
      productRDDResult == productDFResult,
      "Problem 2 results do not match!"
    )

    println("Problem 2: RDD and DataFrame results MATCH")

    // ==========================================
    // 5. PROBLEM 3: ORDER COUNT BY STATUS
    // ==========================================

    println("\n========== PROBLEM 3: RDD ==========")

    val statusRDD = orderRDD
      .map(order =>
        (order.status, 1L)
      )
      .reduceByKey(_ + _)
      .sortBy(
        row => (-row._2, row._1),
        ascending = true,
        numPartitions = sc.defaultParallelism
      )

    val statusRDDResult = statusRDD.collect().toSeq

    statusRDDResult.foreach(println)

    println("\n========== PROBLEM 3: DATAFRAME ==========")

    val statusDF = orderDF
      .groupBy("status")
      .agg(
        count(lit(1)).as("order_count")
      )
      .orderBy(
        col("order_count").desc,
        col("status").asc
      )

    statusDF.show(false)

    val statusDFResult = statusDF.collect().map { row =>
      (
        row.getAs[String]("status"),
        row.getAs[Long]("order_count")
      )
    }.toSeq

    assert(
      statusRDDResult == statusDFResult,
      "Problem 3 results do not match!"
    )

    println("Problem 3: RDD and DataFrame results MATCH")

    // ==========================================
    // 6. RDD LINEAGE INVESTIGATION
    // ==========================================

    println("\n========== RDD LINEAGE ==========")

    println("\nProblem 1 RDD Lineage:")
    println(customerRDD.toDebugString)

    println("\nProblem 2 RDD Lineage:")
    println(productRDD.toDebugString)

    println("\nProblem 3 RDD Lineage:")
    println(statusRDD.toDebugString)

    // ==========================================
    // 7. DATAFRAME PHYSICAL PLANS
    // ==========================================

    println("\n========== DATAFRAME PHYSICAL PLANS ==========")

    println("\nProblem 1 Physical Plan:")
    customerDF.explain("formatted")

    println("\nProblem 2 Physical Plan:")
    productDF.explain("formatted")

    println("\nProblem 3 Physical Plan:")
    statusDF.explain("formatted")

    // ==========================================
    // 8. FINAL VALIDATION
    // ==========================================

    println("\n========== FINAL VALIDATION ==========")

    println("Problem 1: PASSED")
    println("Problem 2: PASSED")
    println("Problem 3: PASSED")

    println("\nALL RDD AND DATAFRAME RESULTS MATCHED!")

    spark.stop()
  }
}
