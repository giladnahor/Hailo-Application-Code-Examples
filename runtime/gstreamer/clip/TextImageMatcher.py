import time
import json
import numpy as np
from PIL import Image
import logging
from logger_setup import setup_logger, set_log_level
from breakpoint_evey_n_frames import set_breakpoint_every_n_frames, update_n_frames

# Set up the logger
logger = setup_logger()
# Change the log level to INFO
set_log_level(logger, logging.INFO)

# Set up global variables. Only required imports are done in the init functions
clip = None
ort = None
torch = None


class TextEmbeddingEntry:
    def __init__(self, text="", embedding=None, negative=False, ensemble=False):
        self.text = text
        self.embedding = embedding if embedding is not None else np.array([])
        self.negative = negative
        self.ensemble = ensemble
        self.probability = 0.0
    def to_dict(self):
        return {
            "text": self.text,
            "embedding": self.embedding.tolist(),  # Convert numpy array to list
            "negative": self.negative,
            "ensemble": self.ensemble
        }

class TextImageMatcher:
    def __init__(self, model_name="RN50x4", threshold=0.8, max_entries=6, global_best=False):
        self.model = None # model is initialized in init_onnx or init_clip
        self.model_runtime = None
        self.model_name = model_name
        self.threshold = threshold
        self.global_best = global_best
        self.entries = [TextEmbeddingEntry() for _ in range(max_entries)]
        self.text_prefix = "A photo of a "
        self.ensemble_template = [
            'a photo of a {}.',
            'a photo of the {}.',
            'a photo of my {}.',
            'a photo of a big {}.',
            'a photo of a small {}.',
        ]
    # Define class as a singleton
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextImageMatcher, cls).__new__(cls)
        return cls._instance

    def init_onnx(self, onnx_model_path):
        global clip, ort
        import clip
        import onnxruntime as ort
        print(f"Loading ONNX model this might take a while...")
        self.model = ort.InferenceSession(onnx_model_path)
        self.model_runtime = "onnx"

    def init_clip(self):
        global clip, torch
        import clip
        import torch
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        print(f"Loading model {self.model_name} on device {device} this might take a while...")
        self.model, self.preprocess = clip.load(self.model_name, device=device)
        self.device = device
        self.model_runtime = "clip"

    def set_threshold(self, new_threshold):
        self.threshold = new_threshold
    
    def set_text_prefix(self, new_text_prefix):
        self.text_prefix = new_text_prefix
    
    def set_ensemble_template(self, new_ensemble_template):
        self.ensemble_template = new_ensemble_template

    def set_global_best(self, new_global_best):
        # if global_best is True, softmax is run on all entries
        # if global_best is False, softmax is run on each row in the image embedding
        self.global_best = new_global_best

    def update_text_entries(self, new_entry, index=None):
        if index is None:
            for i, entry in enumerate(self.entries):
                if entry.text == "":
                    self.entries[i] = new_entry
                    return
            print("Error: Entry list is full.")
        elif 0 <= index < len(self.entries):
            self.entries[index] = new_entry
        else:
            print(f"Error: Index out of bounds: {index}")

    def add_text(self, text, index=None, negative=False, ensemble=False):
        global clip, torch
        if self.model_runtime is None:
            print("Error: No model is loaded. Please call init_onnx or init_clip before calling add_text.")
            return
        if ensemble:
            text_entries = [template.format(text) for template in self.ensemble_template]
        else:
            text_entries = [self.text_prefix + text]
        logger.debug(f"Adding text entries: {text_entries}")
        if self.model_runtime == "onnx":
            text_tokens = clip.tokenize(text_entries)
            # Convert to numpy int64 array, as expected by the model
            text_tokens = text_tokens.numpy().astype(np.int64)
            text_features = self.model.run(None, {'input': text_tokens})[0]
            norm = np.linalg.norm(text_features, axis=-1, keepdims=True)
            text_features /= norm
            ensemble_embedding = np.mean(text_features, axis=0).flatten()
        else:
            text_tokens = clip.tokenize(text_entries).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                ensemble_embedding = torch.mean(text_features, dim=0).cpu().numpy().flatten()
        # new_entry = TextEmbeddingEntry(text, ensemble_embedding.cpu().numpy().flatten(), negative, ensemble)
        new_entry = TextEmbeddingEntry(text, ensemble_embedding, negative, ensemble)
        self.update_text_entries(new_entry, index)

    def get_embeddings(self):
        # return a list of indexes to self.entries if entry.text != ""
        valid_entries = [i for i, entry in enumerate(self.entries) if entry.text != ""]

        return valid_entries

    def get_texts(self):
        # returns all entries text (not only valid ones)
        return [entry.text for entry in self.entries]
    
    def save_embeddings(self, filename):
        with open(filename, 'w') as f:
            json.dump([entry.to_dict() for entry in self.entries], f)

    def load_embeddings(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            self.entries = [TextEmbeddingEntry(text=entry['text'], 
                                               embedding=np.array(entry['embedding']), 
                                               negative=entry['negative'],
                                               ensemble=entry['ensemble']) 
                                               for entry in data]

    def get_image_embedding(self, image):
        if self.model_runtime is None:
            print("Error: No model is loaded. Please call init_clip before calling get_image_embedding.")
            return
        if self.model_runtime == "onnx":
            print("Error: get_image_embedding is not supported for ONNX models.")
            return
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_input)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        return image_embedding.cpu().numpy().flatten()     

    def match(self, image_embedding_np):
        # This function is used to match an image embedding to a text embedding
        # If global_best is True, the best match from all entries is returned
        # Otherwise, the best match from each row in the image embedding is returned
        # Returns a list of tuples: (row_idx, text, similarity)
        # row_idx is the index of the row in the image embedding
        # text is the best matching text
        # similarity is the similarity between the image and text embeddings
        # If the best match is a negative entry, or if the similarity is below the threshold, the tuple is not returned
        # If no match is found, an empty list is returned

        if len(image_embedding_np.shape) == 1:
            image_embedding_np = image_embedding_np.reshape(1, -1)

        results = []
        all_dot_products = None
        # set_breakpoint_every_n_frames()
        valid_entries = self.get_embeddings()
        if len(valid_entries) == 0:
            return []
        text_embeddings_np = np.array([self.entries[i].embedding for i in valid_entries])
        for row_idx, image_embedding_1d in enumerate(image_embedding_np):
            dot_products = np.dot(text_embeddings_np, image_embedding_1d)
            # add dot_products to all_dot_products as new line
            if all_dot_products is None:
                all_dot_products = dot_products[np.newaxis, :]
                
            else:
                all_dot_products = np.vstack((all_dot_products, dot_products))
            
            if not self.global_best:
                # Compute softmax for each row (i.e. each image embedding)
                similarities = np.exp(100 * dot_products)
                similarities /= np.sum(similarities)
                best_idx = np.argmax(similarities)
                best_similarity = similarities[best_idx]
                for i, value in enumerate(similarities):
                    self.entries[valid_entries[i]].probability = similarities[i]
    
                if self.entries[valid_entries[best_idx]].negative:
                    # Background is the best match
                    continue
                if best_similarity > self.threshold:
                    results.append([row_idx, self.entries[valid_entries[best_idx]].text, best_similarity])
        if self.global_best:
            # all_dot_products is a matrix of shape (# image embeddings, # text embeddings)
            # Compute softmax for all entries
            all_dot_products = np.array(all_dot_products)
            similarities = np.exp(500 * all_dot_products)
            similarities /= np.sum(similarities)
            #return argmax over all elements best_idx is a tuple (row_idx, col_idx)
            best_idx = np.unravel_index(np.argmax(similarities, axis=None), similarities.shape)
            best_similarity = similarities[best_idx]
            # update probabilities for the best match i.e. for the row_idx
            for i, value in enumerate(similarities[best_idx[0],:]):
                self.entries[valid_entries[i]].probability = similarities[best_idx[0],i]

            if self.entries[valid_entries[best_idx[1]]].negative:
                # Background is the best match
                return []
            if best_similarity > self.threshold:
                results.append([best_idx[0], self.entries[valid_entries[best_idx[1]]].text, best_similarity])
        logger.debug(f"Best match output: {results}")
        return results
    
if __name__ == "__main__":
    # get cli args
    import argparse
    parser = argparse.ArgumentParser()
    # add onnx_path arg
    parser.add_argument("--onnx", action="store_true", help="use onnx model, requires onnx-path")
    parser.add_argument("--onnx-path", type=str, default="textual.onnx", help="path to onnx model")
    parser.add_argument("--output", type=str, default="text_embeddings.json", help="output file name")
    parser.add_argument("--interactive", action="store_true", help="input text from interactive shell")



    # get args
    args = parser.parse_args()

    # Initialize the matcher and add text embeddings
    matcher = TextImageMatcher()
    if args.onnx:
        matcher.init_onnx(args.onnx_path)
    else:
        matcher.init_clip()
    texts = []
    if args.interactive:
        while True:
            text = input("Enter text ('q' to finish): ")
            if text == "q":
                break
            texts.append(text)
    else:
        texts = [
            "A picture of a cat",
            "A picture of a dog",
            "A photograph of a city",
            "A photo of young people",
            "A photo of old people",
            "A painting of a landscape",
        ]

    print(f"Adding text embeddings: {texts}")
    # Measure the time taken for the add_text function
    start_time = time.time()
    for text in texts:
        matcher.add_text(text)
    end_time = time.time()
    print(f"Time taken to add {len(texts)} text embeddings using add_text(): {end_time - start_time:.4f} seconds")

    matcher.save_embeddings(args.output)

    if args.onnx:
        print("Skipping image embedding generation")
        exit()
    # Read an image from file
    image_path = "people.jpg"
    image = Image.open(image_path)

    # Generate image embedding using the new method
    image_embedding = matcher.get_image_embedding(image)

    # Measure the time taken for the match function
    start_time = time.time()
    result = matcher.match(image_embedding)
    end_time = time.time()

    # Output the results
    print(f"Best match: {result}")
    print(f"Time taken to run match(): {end_time - start_time:.4f} seconds")
