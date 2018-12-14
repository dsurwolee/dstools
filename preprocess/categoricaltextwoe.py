import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from src.preprocess.preprocess_text import PreprocessText
from sklearn.feature_extraction.text import CountVectorizer

class CategoricalTextWoe:
    
    def __init__(self, grid_search={}, analyzer={}, mapping={}, WOE_LIMIT=10):
        self.WOE_UPPER = WOE_LIMIT
        self.WOE_LOWER = -WOE_LIMIT
        self.k_range = grid_search['k_range']
        self.analyzer = analyzer
        self.mapping = mapping
        self.WOE = {}
        self.max_iv = 0
        self.id_max_iv = 0
        self.search_history = {'iteration': [], 'iv': {}, 'woe': {}, 'iv_value': {}, 'params':{}, 'mapping': {}}
        
    def text_analyzer(self, X):
        proceesor = PreprocessText(config=self.analyzer)
        cleaned_X = proceesor.preprocess(X)
        mapping = proceesor.mapping
        return cleaned_X, mapping
        
    def find_rare(self, column, thres=None, rare_values=[]):
        value_dist = column.value_counts(normalize=True)
        rare_values = value_dist[value_dist < thres].to_dict()
        return rare_values

    def replace_rare_values(self, x, rare_values):
        try:
            rare_values[x]
        except:
            return x
        return 'rare'

    def distribution(self, x, y):
        dist = pd.crosstab(x, y, normalize='columns')
        nf = dist.iloc[:,0] 
        f = dist.iloc[:,1]
        return nf, f
    
    def woe(self, x, nf, f):
        woe = np.log(nf/f)
        return woe.clip(self.WOE_LOWER, self.WOE_UPPER)
    
    def information_value(self, cnf, cf, cw):
        return (cnf-cf)*cw
    
    def ngrams(self, X, ngram, analyzer='char', stop_words={'string':'english'}):
        vectorizer = CountVectorizer(analyzer=analyzer, ngram_range=ngram, stop_words=stop_words)
        return vectorizer.fit_transform(X)
    
    def cluster(self, X, k, max_iter=100, n_init=1):
        km = MiniBatchKMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1, init_size=1000, batch_size=1000)
        return km.fit(X) 
    
    def fit(self, X, y, thres=0.001):
        # Clean raw data
        X_cleaned, mapping = self.text_analyzer(X)
        self.mapping = {r: {'clean': c} for r, c in mapping.items()}
        self.X_cleaned = X_cleaned
    
        # Get distinct values of X for clustering
        cX = np.unique(X_cleaned[X_cleaned != 'missing']) 
        self.cX = cX
        cF = self.ngrams(cX, ngram=(3,5))
        
        for k in range(*self.k_range):
            km = self.cluster(cF, k=k)
            labels = pd.Series(km.labels_)
            rare = self.find_rare(labels, thres)

            # Create clean-to-cluster mapping
            csize = cX.shape[0]
            self.cmap = {cX[i]: self.replace_rare_values(labels[i],rare) for i in range(csize)} # Map each clean value to cluster tag,
            self.cmap['missing'] = 'missing'                                                          # missing or rare
            
            # Create raw-to-cluster mapping
            for rv in self.mapping:
                cv = self.mapping[rv]['clean']           # Get cleaned value corresponding to raw value
                self.mapping[rv]['cluster'] = self.cmap[cv]   # Get cluster id corresponding to cleaned value

            # Replace raw value with cluster ID to compute WOE
            X_cluster = X.map(lambda x: self.mapping[x]['cluster'])
            nd, fd = self.distribution(X_cluster, y) 
            woe = self.woe(X_cluster, nd, fd)
            iv = self.information_value(nd, fd, woe)
            iv_total = np.sum(iv)
            
            # Check if IV is higher than current, if so update IV and K
            if iv_total > self.max_iv:
                self.max_iv = iv_total
                self.id_max_iv = k

            # Create raw-to_cluster mapping
            for rv in self.mapping:
                cv = self.mapping[rv]['cluster']
                self.mapping[rv]['woe'] = woe[cv]
                self.mapping[rv]['iv_value'] = iv[cv]

            # Store values
            self.search_history['iteration'].append(k)
            self.search_history['iv'][k] = iv_total
            self.search_history['iv_value'][k] = iv.to_dict()
            self.search_history['woe'][k] = woe.to_dict()
            self.search_history['params'][k] = km.get_params()
            self.search_history['mapping'][k] = self.mapping
            
    def create_table(self, get_max=True, _id=None):
        if get_max:
            _id = self.id_max_iv
        mapping = self.search_history['mapping'][_id]
        iv_table_dict = {}
        for v in mapping:
            value_mapping = mapping[v]
            cluster_id = value_mapping['cluster']
            if cluster_id not in iv_table_dict:
                iv_table_dict[cluster_id] = {'values': [v],'woe': value_mapping['woe'], 'iv': value_mapping['iv_value']}
            else:
                iv_table_dict[cluster_id]['values'].append(v)
        return pd.DataFrame.from_dict(iv_table_dict, orient='index').sort_values('woe')

    def replace(self, x, mapping=None):
        mapping = self.search_history['mapping'][self.id_max_iv]
        rare_value = self.search_history['woe'][self.id_max_iv]['rare']
        return x.map(lambda y: mapping[y]['woe'] if y in mapping else rare_value)
    