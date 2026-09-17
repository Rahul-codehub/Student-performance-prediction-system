from ml_pipeline import train_and_evaluate

if __name__ == '__main__':
    meta = train_and_evaluate()
    print(f"Selected model: {meta['model_name']}")
    print(f"Dataset rows: {meta['dataset_rows']}")
    print(f"Metrics: {meta['selected_metrics']}")
