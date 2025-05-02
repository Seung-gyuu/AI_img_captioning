from model_loader import pipe

def classify_image(image, top_k=3):
    results = pipe(image)

    top_results = results[:top_k]
    return {result['label']: result['score'] for result in top_results}
