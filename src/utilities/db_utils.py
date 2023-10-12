import json
import datetime
import sqlite3
import logging

from config import UserConfiguration, TextGenConfiguration, TorchConfiguration, TokenizerConfiguration, \
    SystemConfiguration, TrainerConfiguration, QuantizationConfiguration, LoraConfiguration


def create_tables(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. God Configurations Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS GodConfigurations (
        GOD_TAG TEXT,
        CONFIG_KEY TEXT,
        CONFIG_VALUE TEXT,
        PRIMARY KEY (GOD_TAG, CONFIG_KEY)
        )
    ''')

# 2. Runs Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Runs (
        RUN_NAME TEXT PRIMARY KEY,
        TIMESTAMP DATETIME,
        CARED_CONFIG_JSON TEXT,
        GOD_TAG TEXT,
        FOREIGN KEY (GOD_TAG) REFERENCES GodConfigurations(GOD_TAG)
        )
    ''')

    # 3. Metrics Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Metrics (
    TIMESTAMP DATETIME,
    METRIC_TAG TEXT,
    RUN_NAME TEXT,
    METRIC_JSON TEXT,
    PRIMARY KEY (TIMESTAMP, METRIC_TAG, RUN_NAME),
    FOREIGN KEY (RUN_NAME) REFERENCES Runs(RUN_NAME)
);
    ''')

    # 4. Checkpoint Paths Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS CheckpointPaths (
    TIMESTAMP DATETIME,
    EPOCH REAL,
    RUN_NAME TEXT,
    PATH TEXT,
    PRIMARY KEY (TIMESTAMP, RUN_NAME, EPOCH),
    FOREIGN KEY (RUN_NAME) REFERENCES Runs(RUN_NAME)
);
    ''')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()


def extract_god_configs_to_flat_dict(tokenizer):
    """Get all configurations as a flat dictionary."""
    configs = {
        'user_config': vars(UserConfiguration()),
        'system_config': vars(SystemConfiguration()),
        'tokenizer_config': vars(TokenizerConfiguration()),
        'torch_config': vars(TorchConfiguration()),
        'text_gen_config': vars(TextGenConfiguration(tokenizer)),
        'train_config': vars(TrainerConfiguration()),
        'quantization_config': vars(QuantizationConfiguration()),
        'lora_config': vars(LoraConfiguration()),
    }

    flat_configs = {}
    for config_name, config_values in configs.items():
        for key, value in config_values.items():
            # Generate a unique key combining the configuration name and the specific key
            flat_key = f"{config_name}.{key}"
            flat_configs[flat_key] = value

    return flat_configs


def store_god_configurations_if_not_exists(db_path, god_tag, tokenizer):
    """Store the god configurations in a flattened way."""
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Extract the configurations as a flat dictionary
    flat_configs = extract_god_configs_to_flat_dict(tokenizer)

    for config_key, value in flat_configs.items():
        # Check if this specific configuration already exists
        cursor.execute(
            'SELECT * FROM GodConfigurations WHERE GOD_TAG = ? AND CONFIG_KEY = ?',
            (god_tag, config_key)
        )
        result = cursor.fetchone()

        # If it doesn't exist, insert it
        if not result:
            cursor.execute(
                'INSERT INTO GodConfigurations (GOD_TAG, CONFIG_KEY, CONFIG_VALUE) VALUES (?, ?, ?)',
                (god_tag, config_key, str(value))  # Convert value to string to ensure it can be stored
            )
    conn.commit()
    conn.close()


def generate_run_name(cared_configurations):
    """Generate an aesthetic run name by concatenating the key-value pairs."""
    return "|".join([f"{key}-{value}" for key, value in cared_configurations.items()])


def store_cared_configurations(db_path, god_tag, cared_configurations):
    """Store the cared configurations in the Runs table."""
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Generate the aesthetic run name
    run_name = generate_run_name(cared_configurations)

    # Current timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Convert the cared configurations dictionary to a JSON string
    cared_config_json = json.dumps(cared_configurations)

    # Insert into the Runs table
    cursor.execute(
        'INSERT INTO Runs (RUN_NAME, TIMESTAMP, CARED_CONFIG_JSON, GOD_TAG) VALUES (?, ?, ?, ?)',
        (run_name, timestamp, cared_config_json, god_tag)
    )
    conn.commit()

    conn.close()


def store_metric(db_path, metric_tag, run_name, metric_details):
    """Store a metric entry in the Metrics table."""
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Current timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"DB_PATH {db_path}, metric_tag {metric_tag}, run_name {run_name}, metric_details {metric_details}, timestamp {timestamp}")

    # Convert the metric details (assuming it's a dictionary) to a JSON string
    metric_json = json.dumps(metric_details)

    # Insert into the Metrics table
    cursor.execute(
        'INSERT INTO Metrics (TIMESTAMP, METRIC_TAG, RUN_NAME, METRIC_JSON) VALUES (?, ?, ?, ?)',
        (timestamp, metric_tag, run_name, metric_json)
    )
    conn.commit()

    conn.close()


def store_checkpoint(db_path, epoch, run_name, path):
    """Store a checkpoint entry in the CheckpointPaths table."""
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Current timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Insert into the CheckpointPaths table
    cursor.execute(
        'INSERT INTO CheckpointPaths (TIMESTAMP, EPOCH, RUN_NAME, PATH) VALUES (?, ?, ?, ?)',
        (timestamp, epoch, run_name, path)
    )
    conn.commit()

    conn.close()

