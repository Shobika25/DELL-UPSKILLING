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


    val spark = SparkSession.builder()
      .appName("RDD vs DataFrame")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._

    val employees = Seq(
      Employee(101, "Ravi", "IT", 80000L),
      Employee(102, "Priya", "IT", 90000L),
      Employee(103, "Arun", "HR", 50000L),
      Employee(104, "Kumar", "HR", 52000L),
      Employee(105, "Sneha", "Sales", 60000L),
      Employee(106, "Divya", "Sales", 65000L),
      Employee(107, "Raj", "Sales", 55000L)
    )

    val employeeRDD =
      spark.sparkContext.parallelize(employees, 2)

    val departmentSalaryRDD =
      employeeRDD.map(e => (e.department, e.salary))

    val totalSalaryRDD =
      departmentSalaryRDD.reduceByKey(_ + _)

    val sortedSalaryRDD =
      totalSalaryRDD.sortBy {
        case (department, total) => (-total, department)
      }

    println("========== RDD OUTPUT ==========")

    sortedSalaryRDD.collect().foreach {
      case (department, total) =>
        println(department + " : " + total)
    }

    val employeeDF = employees.toDF()

    val departmentTotalsDF =
      employeeDF
        .groupBy("department")
        .agg(
          sum("salary").cast("long").as("total_salary")
        )

    val sortedSalaryDF =
      departmentTotalsDF
        .orderBy(desc("total_salary"), asc("department"))

    println("\n========== DATAFRAME OUTPUT ==========")

    sortedSalaryDF.show()
    println("\n========== DATAFRAME EXECUTION PLAN ==========")
    sortedSalaryDF.explain(true)

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

    spark.stop()
  }
}
