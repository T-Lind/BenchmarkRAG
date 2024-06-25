from ragstack_colbert import CassandraDatabase, ColbertEmbeddingModel, ColbertVectorStore
from dotenv import load_dotenv
import logging
import nest_asyncio
from ragstack_colbert import ColbertRetriever
import os
from langchain_community.retrievers import KNNRetriever
from langchain_openai import OpenAIEmbeddings

arctic_botany_dict = {
    "Introduction to Arctic Botany": "Arctic botany is the study of plant life in the Arctic, a region characterized by extreme cold, permafrost, and minimal sunlight for much of the year. Despite these harsh conditions, a diverse range of flora thrives here, adapted to survive with minimal water, low temperatures, and high light levels during the summer. This introduction aims to shed light on the resilience and adaptation of Arctic plants, setting the stage for a deeper dive into the unique botanical ecosystem of the Arctic.",
    "Arctic Plant Adaptations": "Plants in the Arctic have developed unique adaptations to endure the extreme climate. Perennial growth, antifreeze proteins, and a short growth cycle are among the evolutionary solutions. These adaptations not only allow the plants to survive but also to reproduce in short summer months. Arctic plants often have small, dark leaves to absorb maximum sunlight, and some species grow in cushion or mat forms to resist cold winds. Understanding these adaptations provides insights into the resilience of Arctic flora.",
    "The Tundra Biome": "The Arctic tundra is a vast, treeless biome where the subsoil is permanently frozen. Here, the vegetation is predominantly composed of dwarf shrubs, grasses, mosses, and lichens. The tundra supports a surprisingly rich biodiversity, adapted to its cold, dry, and windy conditions. The biome plays a crucial role in the Earth's climate system, acting as a carbon sink. However, it's sensitive to climate change, with thawing permafrost and shifting vegetation patterns.",
    "Arctic Plant Biodiversity": "Despite the challenging environment, the Arctic boasts a significant variety of plant species, each adapted to its niche. From the colorful blooms of Arctic poppies to the hardy dwarf willows, these plants form a complex ecosystem. The biodiversity of Arctic flora is vital for local wildlife, providing food and habitat. This diversity also has implications for Arctic peoples, who depend on certain plant species for food, medicine, and materials.",
    "Climate Change and Arctic Flora": "Climate change poses a significant threat to Arctic botany, with rising temperatures, melting permafrost, and changing precipitation patterns. These changes can lead to shifts in plant distribution, phenology, and the composition of the Arctic flora. Some species may thrive, while others could face extinction. This dynamic is critical to understanding future Arctic ecosystems and their global impact, including feedback loops that may exacerbate global warming.",
    "Research and Conservation in the Arctic": "Research in Arctic botany is crucial for understanding the intricate balance of this ecosystem and the impacts of climate change. Scientists conduct studies on plant physiology, genetics, and ecosystem dynamics. Conservation efforts are focused on protecting the Arctic's unique biodiversity through protected areas, sustainable management practices, and international cooperation. These efforts aim to preserve the Arctic flora for future generations and maintain its role in the global climate system.",
    "Traditional Knowledge and Arctic Botany": "Indigenous peoples of the Arctic have a deep connection with the land and its plant life. Traditional knowledge, passed down through generations, includes the uses of plants for nutrition, healing, and materials. This body of knowledge is invaluable for both conservation and understanding the ecological relationships in Arctic ecosystems. Integrating traditional knowledge with scientific research enriches our comprehension of Arctic botany and enhances conservation strategies.",
    "Future Directions in Arctic Botanical Studies": "The future of Arctic botany lies in interdisciplinary research, combining traditional knowledge with modern scientific techniques. As the Arctic undergoes rapid changes, understanding the ecological, cultural, and climatic dimensions of Arctic flora becomes increasingly important. Future research will need to address the challenges of climate change, explore the potential for Arctic plants in biotechnology, and continue to conserve this unique biome. The resilience of Arctic flora offers lessons in adaptation and survival relevant to global challenges."
}
arctic_botany_texts = list(arctic_botany_dict.values())

load_dotenv()

astra_database_id = os.getenv('ASTRA_DATABASE_ID')
astra_token = os.getenv('ASTRA_TOKEN')
keyspace = "toy1"

database = CassandraDatabase.from_astra(
    astra_token=astra_token,
    database_id=os.getenv('ASTRA_DATABASE_ID'),
    keyspace=keyspace
)

embedding_model = ColbertEmbeddingModel()
# vector_store = ColbertVectorStore(
#     database=database,
#     embedding_model=embedding_model,
# )
#
# results = vector_store.add_texts(texts=arctic_botany_texts, doc_id="arctic_botany")

nest_asyncio.apply()

logging.getLogger('cassandra').setLevel(logging.ERROR)  # workaround to suppress logs

retriever = ColbertRetriever(
    database=database, embedding_model=embedding_model
)

question = "What's happened with artic botany?"
k = 2
answers = retriever.text_search(question, k=k)
for answer in answers:
    print(f"Text: {answer[0].text} Score: {answer[1]}\n")
print("\n--\n\n")
knn_retriever = KNNRetriever.from_texts(arctic_botany_texts, OpenAIEmbeddings(model='text-embedding-3-large'))

knn_answers = knn_retriever.invoke(question, k=knn_retriever, score=True)
for answer in knn_answers:
    print(f"Text: {answer.page_content}")
