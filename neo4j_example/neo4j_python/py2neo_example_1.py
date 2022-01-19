from py2neo import Graph
graph = Graph("bolt://localhost:7687",auth=("neo4j",'qwe123'))

tx = graph.begin()
for name in ["Alice", "Bob", "Carol"]:
    tx.append("CREATE (person:Person name: $name) RETURN person", name=name)
alice, bob, carol = [result.one for result in tx.commit()]

print(alice, bob, carol)