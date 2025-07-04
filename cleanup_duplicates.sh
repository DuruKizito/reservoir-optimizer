#!/bin/bash
# cleanup_duplicates.sh
# This script finds and lists duplicate files in the current directory and subdirectories, then prompts for deletion.

# Find duplicate files by hash
find . -type f -exec sha256sum {} + | sort | uniq -d -w 64 > duplicate_hashes.txt

if [ ! -s duplicate_hashes.txt ]; then
    echo "No duplicate files found."
    rm duplicate_hashes.txt
    exit 0
fi

# List duplicate files
while read hash; do
    grep "^$hash" <(find . -type f -exec sha256sum {} +) | awk '{print $2}'
done < duplicate_hashes.txt > duplicate_files.txt

# Show duplicates and prompt for deletion
cat duplicate_files.txt

echo "\nDo you want to delete all but one copy of each duplicate file? (y/n)"
read answer
if [ "$answer" = "y" ]; then
    while read hash; do
        files=( $(grep "^$hash" <(find . -type f -exec sha256sum {} +) | awk '{print $2}') )
        # Keep the first file, delete the rest
        for ((i=1; i<${#files[@]}; i++)); do
            rm -v "${files[$i]}"
        done
    done < duplicate_hashes.txt
    echo "Duplicate files cleaned up."
else
    echo "No files were deleted."
fi

rm -f duplicate_hashes.txt duplicate_files.txt
