"""
Spark schema definitions for IEEE-CIS fraud detection transactions.

Defines the schema for incoming JSON messages from Kafka.
"""

from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, LongType
)


def get_transaction_schema() -> StructType:
    """
    Returns the Spark schema for incoming transaction JSON.

    This schema covers all IEEE-CIS dataset fields that may appear in
    the incoming Kafka messages.

    Returns:
        StructType: Spark schema for transaction data
    """
    return StructType([
        # Transaction identifiers
        StructField("TransactionID", StringType(), True),
        StructField("TransactionDT", LongType(), True),
        StructField("TransactionAmt", DoubleType(), True),

        # Product
        StructField("ProductCD", StringType(), True),

        # Card information
        StructField("card1", DoubleType(), True),
        StructField("card2", DoubleType(), True),
        StructField("card3", DoubleType(), True),
        StructField("card4", StringType(), True),
        StructField("card5", DoubleType(), True),
        StructField("card6", StringType(), True),

        # Address
        StructField("addr1", DoubleType(), True),
        StructField("addr2", DoubleType(), True),

        # Distance
        StructField("dist1", DoubleType(), True),
        StructField("dist2", DoubleType(), True),

        # Email domains
        StructField("P_emaildomain", StringType(), True),
        StructField("R_emaildomain", StringType(), True),

        # C1-C14 (aggregate features)
        StructField("C1", DoubleType(), True),
        StructField("C2", DoubleType(), True),
        StructField("C3", DoubleType(), True),
        StructField("C4", DoubleType(), True),
        StructField("C5", DoubleType(), True),
        StructField("C6", DoubleType(), True),
        StructField("C7", DoubleType(), True),
        StructField("C8", DoubleType(), True),
        StructField("C9", DoubleType(), True),
        StructField("C10", DoubleType(), True),
        StructField("C11", DoubleType(), True),
        StructField("C12", DoubleType(), True),
        StructField("C13", DoubleType(), True),
        StructField("C14", DoubleType(), True),

        # D1-D15 (timedelta features)
        StructField("D1", DoubleType(), True),
        StructField("D2", DoubleType(), True),
        StructField("D3", DoubleType(), True),
        StructField("D4", DoubleType(), True),
        StructField("D5", DoubleType(), True),
        StructField("D6", DoubleType(), True),
        StructField("D7", DoubleType(), True),
        StructField("D8", DoubleType(), True),
        StructField("D9", DoubleType(), True),
        StructField("D10", DoubleType(), True),
        StructField("D11", DoubleType(), True),
        StructField("D12", DoubleType(), True),
        StructField("D13", DoubleType(), True),
        StructField("D14", DoubleType(), True),
        StructField("D15", DoubleType(), True),

        # M1-M9 (match features - categorical)
        StructField("M1", StringType(), True),
        StructField("M2", StringType(), True),
        StructField("M3", StringType(), True),
        StructField("M4", StringType(), True),
        StructField("M5", StringType(), True),
        StructField("M6", StringType(), True),
        StructField("M7", StringType(), True),
        StructField("M8", StringType(), True),
        StructField("M9", StringType(), True),

        # V1-V339 (Vesta features - only include V1-V120 as per training)
        # Training keeps 120 V-columns, so we include them here
        # For brevity, we'll add V1-V120 dynamically
    ] + [
        StructField(f"V{i}", DoubleType(), True) for i in range(1, 121)
    ] + [
        # Identity fields (from identity table merge)
        StructField("DeviceType", StringType(), True),
        StructField("DeviceInfo", StringType(), True),

        # id_01 - id_38
        StructField("id_01", DoubleType(), True),
        StructField("id_02", DoubleType(), True),
        StructField("id_03", DoubleType(), True),
        StructField("id_04", DoubleType(), True),
        StructField("id_05", DoubleType(), True),
        StructField("id_06", DoubleType(), True),
        StructField("id_07", DoubleType(), True),
        StructField("id_08", DoubleType(), True),
        StructField("id_09", DoubleType(), True),
        StructField("id_10", DoubleType(), True),
        StructField("id_11", DoubleType(), True),
        StructField("id_12", StringType(), True),
        StructField("id_13", DoubleType(), True),
        StructField("id_14", DoubleType(), True),
        StructField("id_15", StringType(), True),
        StructField("id_16", StringType(), True),
        StructField("id_17", DoubleType(), True),
        StructField("id_18", DoubleType(), True),
        StructField("id_19", DoubleType(), True),
        StructField("id_20", DoubleType(), True),
        StructField("id_21", DoubleType(), True),
        StructField("id_22", DoubleType(), True),
        StructField("id_23", StringType(), True),
        StructField("id_24", DoubleType(), True),
        StructField("id_25", DoubleType(), True),
        StructField("id_26", DoubleType(), True),
        StructField("id_27", StringType(), True),
        StructField("id_28", StringType(), True),
        StructField("id_29", StringType(), True),
        StructField("id_30", StringType(), True),
        StructField("id_31", StringType(), True),
        StructField("id_32", DoubleType(), True),
        StructField("id_33", StringType(), True),
        StructField("id_34", StringType(), True),
        StructField("id_35", StringType(), True),
        StructField("id_36", StringType(), True),
        StructField("id_37", StringType(), True),
        StructField("id_38", StringType(), True),

        # Metadata (may be added by producer)
        StructField("timestamp", StringType(), True),
        StructField("ip_address", StringType(), True),
    ])


def get_output_schema() -> StructType:
    """
    Returns the Spark schema for output predictions.

    Returns:
        StructType: Spark schema for prediction results
    """
    return StructType([
        StructField("transaction_id", StringType(), False),
        StructField("fraud_probability", DoubleType(), False),
        StructField("decision", StringType(), False),
        StructField("risk_level", StringType(), False),
        StructField("risk_factors", StringType(), False),  # JSON array as string
        StructField("ato_risk", DoubleType(), False),
        StructField("velocity_risk", DoubleType(), False),
        StructField("amount_risk", DoubleType(), False),
        StructField("timestamp", StringType(), False),
    ])
