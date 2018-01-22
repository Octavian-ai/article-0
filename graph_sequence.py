
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
				(a:PERSON) 
					-[:WROTE]-> 
				(b:REVIEW {dataset_name:{dataset_name}, test:{test}}) 
					-[:OF]-> 
				(c:PRODUCT)
			RETURN a.style_preference + c.style as x, b.score as y
		"""

		self.query_params = {
			"dataset_name": "article_0",
			"test": test
		}

		with open('./settings.json') as f:
			self.settings = json.load(f)

		driver = GraphDatabase.driver(
			self.settings["neo4j_url"], 
			auth=(self.settings["neo4j_user"], self.settings["neo4j_password"]))

		with driver.session() as session:
			data = session.run(self.query, **self.query_params).data()
			data = [ (np.array(i["x"]), i["y"]) for i in data]
			
			# Split the data up into "batches"
			data = more_itertools.chunked(data, self.batch_size)

			# Format our batches in the way Keras expects them
			data = list(data)
			self.data = [ (np.array([j[0] for j in i]), np.array([j[1] for j in i])) for i in data]


	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]
