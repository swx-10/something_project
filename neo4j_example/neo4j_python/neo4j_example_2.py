from neo4j import GraphDatabase


uri = "neo4j://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "qwe123"))

def create_friend_of(tx, name, friend,predicate):
    tx.run(
            # "MATCH (a:Person) WHERE a.name = $name "
            "CREATE (a:Person {name:$name})-[:%s]->(:Person {name: $friend})".format(predicate), name=name, friend=friend)

with driver.session() as session:
    session.write_transaction(create_friend_of, "Alice", "Bob",'friend')

# with driver.session() as session:
#     session.write_transaction(create_friend_of, "Alice", "Carl")

driver.close()