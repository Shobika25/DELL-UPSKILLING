
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

object NestedJSONAssignment {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("Nested JSON Assignment")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val rawDF = spark.read
      .text("src/main/resources/orders.json")

    println("========== RAW JSON RECORDS ==========")
    rawDF.show(false)

    println("========== RAW DATA SCHEMA ==========")
    rawDF.printSchema()

    val customerSchema = StructType(Seq(
      StructField("id", StringType, true),
      StructField("name", StringType, true),
      StructField("city", StringType, true)
    ))

    val itemSchema = ArrayType(
      StructType(Seq(
        StructField("product", StringType, true),
        StructField("quantity", IntegerType, true),
        StructField("price", DoubleType, true)
      )),
      true
    )

    val paymentSchema = StructType(Seq(
      StructField("mode", StringType, true),
      StructField("status", StringType, true)
    ))

    val orderSchema = StructType(Seq(
      StructField("orderId", StringType, true),
      StructField("orderDate", StringType, true),
      StructField("customer", customerSchema, true),
      StructField("items", itemSchema, true),
      StructField("payment", paymentSchema, true)
    ))

    println("========== EXPLICIT ORDER SCHEMA ==========")
    println(orderSchema.treeString)

    val parsedDF = rawDF.select(
      from_json(col("value"), orderSchema).as("order")
    )

    println("========== PARSED JSON ==========")
    parsedDF.show(false)
    parsedDF.printSchema()

    val ordersDF = parsedDF.select(
      col("order.orderId").as("orderId"),
      col("order.orderDate").as("orderDate"),
      col("order.customer.id").as("customerId"),
      col("order.customer.name").as("customerName"),
      col("order.customer.city").as("city"),
      col("order.items").as("items"),
      col("order.payment.mode").as("paymentMode"),
      col("order.payment.status").as("paymentStatus")
    )

    println("========== EXTRACTED ORDER DATA ==========")
    ordersDF.show(false)

    val productsDF = ordersDF
      .withColumn("item", explode(col("items")))
      .select(
        col("orderId"),
        col("orderDate"),
        col("customerId"),
        col("customerName"),
        col("city"),
        col("item.product").as("product"),
        col("item.quantity").as("quantity"),
        col("item.price").as("price"),
        col("paymentMode"),
        col("paymentStatus")
      )
      .withColumn(
        "productAmount",
        col("quantity") * col("price")
      )

    println("========== EXPLODED PRODUCT DATA ==========")
    productsDF.show(false)

    productsDF.printSchema()

    val productSalesDF = productsDF
      .groupBy("product")
      .agg(
        sum("quantity").as("totalQuantity"),
        sum("productAmount").as("totalSales")
      )
      .orderBy(desc("totalSales"))

    println("========== PRODUCT SALES ==========")
    productSalesDF.show(false)


    val orderTotalsDF = productsDF
      .groupBy(
        "orderId",
        "orderDate",
        "customerId",
        "customerName",
        "city",
        "paymentMode",
        "paymentStatus"
      )
      .agg(
        sum("productAmount").as("orderTotal")
      )

    println("========== ORDER TOTALS ==========")
    orderTotalsDF.orderBy("orderId").show(false)

    ordersDF.createOrReplaceTempView("orders_view")
    productsDF.createOrReplaceTempView("products_view")
    orderTotalsDF.createOrReplaceTempView("order_totals_view")

    println("========== HYDERABAD ORDERS ==========")

    spark.sql("""
      SELECT orderId, customerName, city
      FROM orders_view
      WHERE city = 'Hyderabad'
    """).show(false)

    println("========== SUCCESSFUL PAYMENTS ==========")

    spark.sql("""
      SELECT orderId, paymentMode, paymentStatus
      FROM orders_view
      WHERE paymentStatus = 'Success'
    """).show(false)

    println("========== PRODUCTS WITH PRICE ABOVE 10000 ==========")

    spark.sql("""
      SELECT DISTINCT product, price
      FROM products_view
      WHERE price > 10000
    """).show(false)

    println("========== TOTAL SPENDING BY CUSTOMER ==========")

    spark.sql("""
      SELECT customerName,
             SUM(orderTotal) AS totalSpending
      FROM order_totals_view
      GROUP BY customerName
      ORDER BY totalSpending DESC
    """).show(false)

    val latestOrderWindow = Window
      .partitionBy("customerId")
      .orderBy(desc("orderDate"))

    val latestOrdersDF = orderTotalsDF
      .withColumn(
        "row_number",
        row_number().over(latestOrderWindow)
      )
      .filter(col("row_number") === 1)

    println("========== LATEST ORDER PER CUSTOMER ==========")
    latestOrdersDF.show(false)

    val orderValueWindow = Window
      .partitionBy("customerId")
      .orderBy(desc("orderTotal"))

    val rankedOrdersDF = orderTotalsDF
      .withColumn(
        "orderRank",
        rank().over(orderValueWindow)
      )

    println("========== RANKED ORDERS ==========")
    rankedOrdersDF.orderBy("customerId", "orderRank").show(false)


    val productRankWindow = Window
      .orderBy(desc("totalSales"))

    val rankedProductsDF = productSalesDF
      .withColumn(
        "salesDenseRank",
        dense_rank().over(productRankWindow)
      )

    println("========== PRODUCT DENSE RANK ==========")
    rankedProductsDF.show(false)

    val runningSpendWindow = Window
      .partitionBy("customerId")
      .orderBy("orderDate")
      .rowsBetween(
        Window.unboundedPreceding,
        Window.currentRow
      )

    val runningSpendDF = orderTotalsDF
      .withColumn(
        "runningSpend",
        sum("orderTotal").over(runningSpendWindow)
      )

    println("========== RUNNING CUSTOMER SPENDING ==========")
    runningSpendDF
      .orderBy("customerId", "orderDate")
      .show(false)

    val previousOrderDF = orderTotalsDF
      .withColumn(
        "previousOrderAmount",
        lag("orderTotal", 1).over(
          Window
            .partitionBy("customerId")
            .orderBy("orderDate")
        )
      )

    println("========== PREVIOUS ORDER AMOUNT USING LAG ==========")
    previousOrderDF
      .orderBy("customerId", "orderDate")
      .show(false)

    val secondOrderDF = orderTotalsDF
      .withColumn(
        "orderNumber",
        row_number().over(orderValueWindow)
      )
      .filter(col("orderNumber") === 2)

    println("========== SECOND ORDER BY VALUE PER CUSTOMER ==========")
    secondOrderDF.show(false)

    println("========== TOTAL SALES VALIDATION ==========")

    productsDF
      .agg(sum("productAmount").as("grandTotalSales"))
      .show(false)

    spark.stop()
  }
}
