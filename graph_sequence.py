
import keras
import numpy as np
import more_itertools
import json
from neo4j.v1 import GraphDatabase, Driver

class GraphSequence(keras.utils.Sequence):

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

		with open('./settings.json') as f:
			self.settings = json.load(f)

		driver = GraphDatabase.driver(
			self.settings["neo4j_url"], 
			auth=(self.settings["neo4j_user"], self.settings["neo4j_password"]))

		with driver.session() as session:
			data = session.run(self.query, **self.query_params).data()
			data = [np.flatten([ i["style"], i["style_preference"], [i["score"]] ]) for i in data]
			data = more_itertools.chunked(data, self.batch_size)
			self.data = np.array(list(data))

			print(self.data)


	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]
