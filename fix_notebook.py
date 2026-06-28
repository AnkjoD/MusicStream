import json

notebook_path = r"c:\Users\Ankkun\Documents\lap_trinh\my_project\streamlify\src\notebooks\main.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the cell defining build_user_features_test
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "def build_user_features_test" in "".join(cell.get("source", [])):
        source = cell["source"]
        
        new_source = []
        for line in source:
            if "count(when(col(\"page\") == \"Error\", True)).alias(\"error_visits\")" in line:
                # Add the missing comma and other features
                new_source.append("        count(when(col(\"page\") == \"Error\", True)).alias(\"error_visits\"),\n")
                new_source.append("        count(when(col(\"page\").isin(\"Add Friend\", \"Add to Playlist\"), True)).alias(\"social_actions\"),\n")
            elif "user_features = user_features.withColumn(" in line and "dislike_ratio" in line:
                # We'll replace the derived columns logic below this
                pass
            else:
                new_source.append(line)
                
        # Now we need to carefully replace the derived columns
        # Let's just find the exact block and replace it
        source_str = "".join(new_source)
        
        # Replace the derived columns block
        old_block = '''    user_features = user_features.withColumn(
        "dislike_ratio",
        when(col("total_songs") > 0, col("thumbs_down") / col("total_songs")).otherwise(0.0)
    ).withColumn(
        "avg_sessions_per_day",
        when(col("days_active") > 0, col("total_sessions") / col("days_active")).otherwise(0.0)
    )'''
        
        new_block = '''    user_features = user_features.withColumn(
        "dislike_ratio",
        when(col("total_songs") > 0, col("thumbs_down") / col("total_songs")).otherwise(0.0)
    ).withColumn(
        "avg_sessions_per_day",
        when(col("days_active") > 0, col("total_sessions") / col("days_active")).otherwise(0.0)
    ).withColumn(
        "error_rate",
        when(col("total_sessions") > 0, col("error_visits") / col("total_sessions")).otherwise(0.0)
    ).withColumn(
        "like_dislike_ratio",
        when(col("thumbs_down") > 0, col("thumbs_up") / col("thumbs_down")).otherwise(col("thumbs_up"))
    )'''
        
        source_str = source_str.replace(old_block, new_block)
        
        # Write back to list of lines
        cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in source_str.split("\n")][:-1]
        
        break

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Modified main.ipynb successfully.")
