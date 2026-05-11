#!/bin/sh

set -e

# The baseline configuration file
BASELINE="anemoi_recipe.yaml"

if [ ! -f "$BASELINE" ]; then
    echo "🚨 Error: Cannot find $BASELINE in the current directory."
    exit 1
fi

create_split() {
    SPLIT_NAME=$1
    START_DATE=$2
    END_DATE=$3

    echo "=================================================="
    echo "🛠️  Creating $SPLIT_NAME dataset..."
    echo "📅 Start: $START_DATE | End: $END_DATE"

    TEMP_YAML="${SPLIT_NAME}_recipe.yaml"
    OUTPUT_ZARR="${SPLIT_NAME}_dataset.zarr"

    # Use sed to search for the "start:" and "end:" lines and replace the dates
    # (This assumes your baseline YAML has exactly "start: '...'" and "end: '...'")
    sed -e "1,6s/start: .*/start: '${START_DATE}'/" \
        -e "1,6s/end: .*/end: '${END_DATE}'/" \
        "$BASELINE" > "$TEMP_YAML"

    # Run the anemoi command using the new temporary recipe
    echo "🚀 Running anemoi-datasets create..."
    anemoi-datasets create "$TEMP_YAML" "$OUTPUT_ZARR"

    echo "✅ Finished building $OUTPUT_ZARR!"
    echo "=================================================="
}

# 1. Training Set (8 Years: 1851 - 1858)
create_split "train" "1851-01-01T06:00:00" "1859-01-01T00:00:00"

# 2. Validation Set (1 Year: 1859)
create_split "val" "1859-01-01T06:00:00" "1860-01-01T00:00:00"

# 3. Testing Set (1 Year: 1860)
create_split "test" "1860-01-01T06:00:00" "1861-01-01T00:00:00"

echo "🎉 All datasets created successfully!"
