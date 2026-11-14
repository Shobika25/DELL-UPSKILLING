package org.jetbrains.scala


import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object RDDvsDataFrameShuffleActivity {

  case class Employee(
                       employeeId: Int,
                       name: String,
                       department: String,
                       salary: Long
                     )

  def main(args: Array[String]): Unit = {

    // ==================================
    // STEP 1: CREATE SPARK SESSION
    // ==================================

    val spark = SparkSession.builder()
      .appName("RDD vs DataFrame")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._

    // ==================================
    // STEP 2: CREATE EMPLOYEE DATASET
    // ==================================

    val employees = Seq(
      Employee(101, "Ravi", "IT", 80000L),
      Employee(102, "Priya", "IT", 90000L),
      Employee(103, "Arun", "HR", 50000L),
      Employee(104, "Kumar", "HR", 52000L),
      Employee(105, "Sneha", "Sales", 60000L),
      Employee(106, "Divya", "Sales", 65000L),
      Employee(107, "Raj", "Sales", 55000L)
    )

    // ==================================
    // APPROACH 1: USING RDD
    // ==================================

    // Convert employee collection into RDD
    val employeeRDD =
      spark.sparkContext.parallelize(employees, 2)

    // Extract department and salary
    val departmentSalaryRDD =
      employeeRDD.map(e => (e.department, e.salary))

    // SHUFFLE POINT 1: reduceByKey
    // Combine salaries belonging to the same department
    val totalSalaryRDD =
      departmentSalaryRDD.reduceByKey(_ + _)

    // SHUFFLE POINT 2: sortBy
    // Sort department totals in descending order
    val sortedSalaryRDD =
      totalSalaryRDD.sortBy {
        case (department, total) => (-total, department)
      }

    println("========== RDD OUTPUT ==========")

    sortedSalaryRDD.collect().foreach {
      case (department, total) =>
        println(department + " : " + total)
    }

    // ==================================
    // APPROACH 2: USING DATAFRAME
    // ==================================

    // Convert employee collection into DataFrame
    val employeeDF = employees.toDF()

    // SHUFFLE POINT 1: groupBy + sum
    // Calculate total salary department-wise
    val departmentTotalsDF =
      employeeDF
        .groupBy("department")
        .agg(
          sum("salary").cast("long").as("total_salary")
        )

    // SHUFFLE POINT 2: orderBy
    // Sort total salary in descending order
    val sortedSalaryDF =
      departmentTotalsDF
        .orderBy(desc("total_salary"), asc("department"))

    println("\n========== DATAFRAME OUTPUT ==========")

    sortedSalaryDF.show()
    println("\n========== DATAFRAME EXECUTION PLAN ==========")
    sortedSalaryDF.explain(true)
    // ==================================
    // STEP 3: VERIFY BOTH OUTPUTS
    // ==================================

    val rddResult =
      sortedSalaryRDD.collect().toSeq

    val dfResult =
      sortedSalaryDF.collect().map { row =>
        (
          row.getAs[String]("department"),
          row.getAs[Long]("total_salary")
        )
      }.toSeq

    assert(rddResult == dfResult)

    println("SUCCESS: Both outputs match!")

    // ==================================
    // STEP 4: STOP SPARK SESSION
    // ==================================

    spark.stop()
  }
}
