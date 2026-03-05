"""
Dataset Exploration & Statistical Analysis Script
Analyzes KITTI dataset and generates statistics and visualizations
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def load_kitti_labels(train_label_path):
    """
    Load and parse KITTI format labels
    
    Args:
        train_label_path: Path to training labels directory
        
    Returns:
        DataFrame with all annotations
    """
    all_objects = []

    for label_file in os.listdir(train_label_path):
        if not label_file.endswith(".txt"):
            continue
            
        file_path = os.path.join(train_label_path, label_file)
        
        with open(file_path, "r") as f:
            lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                
                if len(parts) < 8:
                    continue
                
                obj_class = parts[0]
                truncated = float(parts[1])
                occluded = int(parts[2])
                
                xmin = float(parts[4])
                ymin = float(parts[5])
                xmax = float(parts[6])
                ymax = float(parts[7])
                
                all_objects.append({
                    "image": label_file.replace(".txt", ""),
                    "class": obj_class,
                    "truncated": truncated,
                    "occluded": occluded,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                })
    
    return pd.DataFrame(all_objects)


def add_bbox_features(df):
    """Add bounding box derived features"""
    df["bbox_width"] = df["xmax"] - df["xmin"]
    df["bbox_height"] = df["ymax"] - df["ymin"]
    df["bbox_area"] = df["bbox_width"] * df["bbox_height"]
    return df


def print_dataset_statistics(df_labels, num_images):
    """Print basic dataset statistics"""
    stats = {
        "Total Images": num_images,
        "Total Annotations": len(df_labels),
        "Average Objects per Image": round(len(df_labels) / num_images, 2) if num_images > 0 else 0
    }
    
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("="*50 + "\n")
    
    return pd.DataFrame(stats, index=["Value"])


def plot_class_distribution(df_labels, output_dir=None):
    """Plot object class distribution"""
    class_counts = df_labels["class"].value_counts()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    
    plt.title("Object Class Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "class_distribution.png"), bbox_inches='tight', dpi=300)
        print(f"Saved: class_distribution.png")
    
    plt.close()


def plot_bbox_width_vs_height(df_labels, output_dir=None):
    """Plot bounding box width vs height scatter plot"""
    plt.figure(figsize=(10, 8))
    
    sns.scatterplot(
        data=df_labels,
        x="bbox_width",
        y="bbox_height",
        hue="class",
        alpha=0.6,
        s=50
    )
    
    plt.title("Bounding Box Width vs Height", fontsize=14, fontweight='bold')
    plt.xlabel("Width (pixels)", fontsize=12)
    plt.ylabel("Height (pixels)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "bbox_width_vs_height.png"), bbox_inches='tight', dpi=300)
        print(f"Saved: bbox_width_vs_height.png")
    
    plt.close()


def plot_occlusion_distribution(df_labels, output_dir=None):
    """Plot occlusion level distribution"""
    plt.figure(figsize=(8, 5))
    
    sns.countplot(data=df_labels, x="occluded", palette="viridis")
    
    plt.title("Occlusion Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Occluded Level (0=visible, 1=partly, 2=largely, 3=unknown)", fontsize=10)
    plt.ylabel("Count", fontsize=12)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "occlusion_distribution.png"), bbox_inches='tight', dpi=300)
        print(f"Saved: occlusion_distribution.png")
    
    plt.close()


def plot_truncation_distribution(df_labels, output_dir=None):
    """Plot truncation distribution"""
    plt.figure(figsize=(8, 5))
    
    sns.histplot(df_labels["truncated"], bins=20, kde=True, color="coral")
    
    plt.title("Truncation Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Truncated (0=non-truncated, 1=truncated)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "truncation_distribution.png"), bbox_inches='tight', dpi=300)
        print(f"Saved: truncation_distribution.png")
    
    plt.close()


def plot_bbox_area_by_class(df_labels, output_dir=None):
    """Plot bounding box area distribution by class"""
    plt.figure(figsize=(12, 6))
    
    sns.boxplot(
        data=df_labels,
        x="class",
        y="bbox_area",
        palette="Set2"
    )
    
    plt.title("Bounding Box Area by Class", fontsize=14, fontweight='bold')
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Bounding Box Area (pixels²)", fontsize=12)
    plt.xticks(rotation=45)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "bbox_area_by_class.png"), bbox_inches='tight', dpi=300)
        print(f"Saved: bbox_area_by_class.png")
    
    plt.close()


def print_class_statistics(df_labels):
    """Print detailed statistics per class"""
    print("\n" + "="*50)
    print("CLASS-WISE STATISTICS")
    print("="*50)
    
    class_stats = df_labels.groupby("class").agg({
        "bbox_area": ["mean", "std", "min", "max"],
        "truncated": "mean",
        "occluded": "mean",
        "image": "count"
    }).round(2)
    
    class_stats.columns = ["Area_Mean", "Area_Std", "Area_Min", "Area_Max", 
                           "Avg_Truncation", "Avg_Occlusion", "Count"]
    
    print(class_stats)
    print("="*50 + "\n")
    
    return class_stats


def main():
    """Main execution function"""
    # Setup paths - adjust to workspace structure
    project_root = Path(__file__).parent.parent
    train_image_path = project_root / "data" / "raw" / "KITTI" / "training" / "image_2"
    train_label_path = project_root / "data" / "raw" / "KITTI" / "training" / "label_2"
    output_dir = project_root / "experiments" / "visualization"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading data from: {train_label_path}")
    print(f"Output directory: {output_dir}\n")
    
    # Get image files
    image_files = [f for f in os.listdir(train_image_path) 
                   if f.endswith((".png", ".jpg", ".jpeg"))]
    
    print(f"Found {len(image_files)} images")
    
    # Load and parse labels
    print("Loading labels...")
    df_labels = load_kitti_labels(train_label_path)
    print(f"Loaded {len(df_labels)} annotations")
    
    # Add bounding box features
    df_labels = add_bbox_features(df_labels)
    
    # Print basic statistics
    stats_df = print_dataset_statistics(df_labels, len(image_files))
    
    # Print class-wise statistics
    class_stats = print_class_statistics(df_labels)
    
    # Save statistics to CSV
    stats_csv_path = output_dir / "dataset_statistics.csv"
    class_stats.to_csv(stats_csv_path)
    print(f"Saved statistics to: {stats_csv_path}\n")
    
    # Generate visualizations
    print("Generating visualizations...")
    print("-" * 50)
    
    plot_class_distribution(df_labels, output_dir)
    plot_bbox_width_vs_height(df_labels, output_dir)
    plot_occlusion_distribution(df_labels, output_dir)
    plot_truncation_distribution(df_labels, output_dir)
    plot_bbox_area_by_class(df_labels, output_dir)
    
    print("-" * 50)
    print(f"\nAnalysis complete! All visualizations saved to: {output_dir}")
    
    return df_labels, class_stats


if __name__ == "__main__":
    # Set plot style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # Run analysis
    df_labels, class_stats = main()
