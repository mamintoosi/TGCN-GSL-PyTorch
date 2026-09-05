from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from keybert import KeyBERT
import re
import numpy as np

def clean_keyword(keyword):
    """Clean and normalize keywords by removing special characters and extra spaces"""
    keyword = re.sub(r'^[►•]+', '', keyword).strip()
    keyword = re.sub(r'^[-]+', '', keyword).strip()
    return ' '.join(keyword.lower().split())

def extract_keywords(ris_file, min_count=1, min_length=3):
    """Extract and sort keywords by frequency, with improved filtering and abbreviation expansion"""
    keywords = defaultdict(int)
    raw_keywords_count = 0 
    cleaned_keywords_count = 0  

    # Add more generic or unwanted terms here
    common_words = {
        'study', 'analysis', 'research', 'paper', 'article', 'review','technology', 'methodology','smart', 'digital', 'industry', 'innovation', 'system', 'systems', 'approach', 'framework', 'health', 'healthcare', 'fault', 'faults', 'faulty', 'fault-tolerant', 'fault tolerance', 'fault detection', 'fault diagnosis', 'application', 'applications', 
        'applications', 'advancements', 'development', 'method',
        'platform', 'utility', 'trial design', 'study design', 'research methods', 'research synthesis',
        'effectiveness', 'education', 'educational attainment', 'expertise', 
        'implementation research', 'basic and applied research',
        'new approach methodology', 'research agenda', 'science-technology linkages', 'transmission',
        'technology', 'accessible', 'internet of things', 'well-being', 
        'net goodness analysis', 'speed', 'time', 'trajectories', 'trend forecasting', 'trends',
        'trend analysis', 'future directions', 'paradigm shift', 'utility', 'metrics', 'simulation',
        'model averaging', 'longitudinal', 'longitudinal study', 'longitudinal birth cohort', 'birth cohort',
        'cohort', 'cohort study', 'population-based', 'population structure', 'population stratification',
        'multi-ancestry', 'continental ancestry', 'diversity', 'ethnicity', 'race', 'race disparities',
        'effective population size', 'class imbalance', 'administrative data', 
        'high-throughput sequencing', 'long-read sequencing', 'omics data', 'data integration','and',
        'or', 'not', 'the', 'a', 'an', 'of', 'in', 'on', 'for', 'to', 'with',
        'at', 'by', 'as', 'from', 'this', 'that', 'it', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'which', 'who', 'whom', 'what', 'where', 'when',
        'ami', 'aodv', 'apsd', 'avvo', 'ciot', 'cir', 'crown', 'daa', 'dall e', 'dba', 'ddos', 'devops', 'dlt', 'ebu', 'eosc', 'fronthaul', 'gan', 'h-iot', 'hadoop', 'history', 'human ai', 'iiot', 'iomt', 'iot', 'isa-95', 'issa', 'its', 'lpwans', 'lte-u', 'manet', 'mec', 'mimo', 'nas', 'nfv', 'ng-ran', 'ngap', 'nids', 'ntma', 'o-ran', 'offensive ai', 'openai', 'rhm', 'saudi arabia', 'spatial', 'state-of-the-art', 'stc', 'stride', 'swat', 'swot', 'tim', 'tows', 'uas', 'uav', 'uavs', 'vanet', 'vanets', 'vosviewer', 'wban',  'xai', 'z-score',
        'comp', 'ppp'
    }
    # Abbreviation mapping: abbreviation -> full form or higher-level concept
    abbr_map = {
    'ann': 'artificial neural network',
    'cnn': 'convolutional neural network',
    'rnn': 'recurrent neural network',
    'lstm': 'long short-term memory',
    'gru': 'gated recurrent unit',
    'gnn': 'graph neural network',
    'gan': 'generative adversarial network',
    'svm': 'support vector machine',
    'rf': 'random forest',
    'dt': 'decision tree',
    'mlp': 'multi-layer perceptron',
    'pca': 'principal component analysis',
    'nlp': 'natural language processing',
    'ai': 'artificial intelligence',
    'ml': 'machine learning',
    'dl': 'deep learning',
    'drl': 'deep reinforcement learning',
    'fl': 'federated learning',
    'rl': 'reinforcement learning',
    'nn': 'neural network',
    'svr': 'support vector regression',
    'arima': 'autoregressive integrated moving average',
    'knn': 'k-nearest neighbors',
    'hmm': 'hidden markov model',
    'bpn': 'backpropagation neural network',
    'elm': 'extreme learning machine',
    'ga': 'genetic algorithm',
    'pso': 'particle swarm optimization',
    't-s': 'takagi-sugeno',
    'kg': 'knowledge graph'
    }    

    with open(ris_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('KW  - '):
                for kw in line[6:].strip().split('; '):
                    raw_keywords_count += 1  # شمارش کلمات خام
                    cleaned_kw = clean_keyword(kw)
                    
                    # Expand abbreviations if present in abbr_map
                    if cleaned_kw in abbr_map:
                        cleaned_kw = abbr_map[cleaned_kw]
                    
                    # Filter out unwanted terms and abbreviations
                    if (cleaned_kw and len(cleaned_kw) >= min_length
                        and cleaned_kw not in common_words):
                        keywords[cleaned_kw] += 1
                        cleaned_keywords_count += 1  # شمارش کلمات معتبر
    
    # چاپ آمار پردازش
    print("\n=== Keyword Processing Report ===")
    print(f"\n[Filter Settings]")
    print(f"- Minimum length: {min_length} characters")
    print(f"- Minimum count: {min_count} occurrence(s)")
    print(f"- Common words filter: {len(common_words)} terms")
    print(f"- Abbreviations mapping: {len(abbr_map)} terms")
    
    print("\n[Processing Results]")
    print(f"1. Raw keywords extracted: {raw_keywords_count}")
    print(f"2. After cleaning/filtering: {cleaned_keywords_count}")
    print(f"3. Percentage kept: {cleaned_keywords_count/raw_keywords_count:.1%}")
    print(f"4. Unique keywords remaining: {len(keywords)}")
    
    print("\n[Filter Effectiveness]")
    removed_count = raw_keywords_count - cleaned_keywords_count
    print(f"- Total removed: {removed_count} ({removed_count/raw_keywords_count:.1%})")
    print(f"- Common words filtered: {len(common_words)} terms")
    print(f"- Abbreviations expanded: {len(abbr_map)} terms")
    print("="*50)
    
    return dict(sorted(keywords.items(), key=lambda x: (-x[1], x[0])))

from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

def enhanced_clustering(keywords_list, max_clusters=50):
    """Improved clustering using Spectral Clustering"""
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(keywords_list)
    
    # Create affinity matrix using cosine similarity
    affinity_matrix = cosine_similarity(embeddings)
    
    # Configure Spectral Clustering
    spectral = SpectralClustering(
        n_clusters=max_clusters,
        affinity='precomputed',
        random_state=42,
        assign_labels='discretize'
    )
    
    labels = spectral.fit_predict(affinity_matrix)
    
    # Organize results into clusters
    clusters = defaultdict(list)
    for kw, label in zip(keywords_list, labels):
        clusters[label].append(kw)
        
    return clusters


def embedding_based_cluster_naming(clusters, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    """
    Assign a representative label to each cluster based on semantic embeddings.
    The label is chosen as the keyword whose embedding is closest to the cluster centroid.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(model_name)
    named_clusters = {}

    for cluster_id, keywords in clusters.items():
        if not keywords:
            continue
        embeddings = model.encode(keywords)
        centroid = np.mean(embeddings, axis=0)
        # Compute cosine similarity to centroid
        sims = np.dot(embeddings, centroid) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(centroid) + 1e-10)
        best_idx = np.argmax(sims)
        representative = keywords[best_idx]
        named_clusters[representative] = keywords

    return named_clusters

def save_full_report(clusters, filename):
    """Save complete cluster report with all keywords"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("COMPLETE CLUSTER REPORT\n")
        f.write("======================\n\n")
        
        for i, (name, keywords) in enumerate(sorted(clusters.items(), 
                                                 key=lambda x: -len(x[1])), 1):
            f.write(f"CLUSTER {i}: {name}\n")
            f.write(f"Keywords ({len(keywords)}):\n")
            
            # Group keywords with line breaks for readability
            lines = []
            current_line = []
            
            for kw in sorted(keywords):
                if len(', '.join(current_line + [kw])) > 100:
                    lines.append(', '.join(current_line))
                    current_line = []
                current_line.append(kw)
            
            if current_line:
                lines.append(', '.join(current_line))
            
            f.write('\n'.join(lines))
            f.write("\n\n")

from translate import Translator

def translate_to_persian(text):
    try:
        translator = Translator(from_lang="en", to_lang="fa")
        return translator.translate(text)
    except Exception as e:
        print(f"Error: {e}")
        return text
    

def main():
    """Enhanced main execution pipeline"""
    input_file = 'data/scopus.ris' 
    output_base = 'enhanced_keyword_clusters'
    
    print("Loading and processing keywords...")
    keywords = extract_keywords(input_file)
    keywords_list = list(keywords.keys())
    
    top_keywords = set(list(keywords.keys())[:10])  # Top 10 keywords
    print("Top keywords:", top_keywords)
    print("\nPerforming enhanced clustering...")
    clusters = enhanced_clustering(keywords_list, max_clusters=60)
    named_clusters = embedding_based_cluster_naming(clusters)
    
    print("\nFinal Clusters:")
    for name, kws in sorted(named_clusters.items(), key=lambda x: -len(x[1])):
        persian_name = translate_to_persian(name)
        print(f"- {name}, {len(kws)} keywords - {persian_name} ")

    # Save complete results
    save_full_report(named_clusters, f"{output_base}_full_report.txt")
    
    # Save VOSviewer files
    with open(f"{output_base}_thesaurus.txt", 'w', encoding='utf-8') as f:
        f.write("label\treplace by\n")
        for name, kws in named_clusters.items():
            for kw in kws:
                f.write(f"{kw}\t{name}\n")
    
    print("\nResults saved to:")
    print(f"- {output_base}_full_report.txt (Complete cluster details)")
    print(f"- {output_base}_thesaurus.txt (VOSviewer format)")

if __name__ == "__main__":
    main()