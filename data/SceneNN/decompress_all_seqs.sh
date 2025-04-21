#!/bin/bash

sequences=("015" "030" "054" "080" "089" "093" "096" "243" "263" "322")

for seq in "${sequences[@]}"; do
    echo "Begin to extract: $seq"
    ./playback ./${seq}.oni ./$seq
    echo "Finished extracting $seq"
done

echo "Finished extracting all sequences!!!"

