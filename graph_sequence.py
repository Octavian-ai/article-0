
import keras
from neo4j.v1 import GraphDatabase, Driver

class GraphSequence(keras.Sequence):

	def __init__(self, batch_size=32, test=False):
        self.batch_size = batch_size
        
        self.query = """
        	MATCH p=
				(a:PERSON {dataset_name:{dataset_name}, test:{test}}) 
					-[:WROTE]-> 
				(b:REVIEW ) 
					-[:OF]-> 
				(c:PRODUCT)
			RETURN a.style_preference AS style_preference, c.style AS style, b.score AS score
        """

        self.query_params = {
        	"dataset_name": "article0",
        	"test": test
        }

        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as session:
        	ret = session.run(query, **query_params)
        	self.data = more_itertools.chunked(ret.data(), self.batch_size)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
