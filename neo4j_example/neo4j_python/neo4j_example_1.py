from neo4j import GraphDatabase

class HelloWorldExample:

    def __init__(self, uri, user, password):
        # 加载驱动连接neo4j
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # 通过驱动关闭连接
        self.driver.close()

    def print_greeting(self, message):
        # 创建一个会话对象
        with self.driver.session() as session:
            greeting = session.write_transaction(self._create_and_return_greeting, message)
            print(greeting)

    @staticmethod
    def _create_and_return_greeting(tx, message):
        # 回调方法，通过transaction(事务)运行一个Cypher指令：创建一个节点，并返回节点的描述信息
        result = tx.run("CREATE (a:Greeting) "
                        "SET a.message = $message "
                        "RETURN a.message + ', from node ' + id(a)", message=message)
        return result.single()[0]


if __name__ == "__main__":
    greeter = HelloWorldExample("bolt://localhost:7687", "neo4j", "qwe123")
    greeter.print_greeting("hello, world")
    greeter.close()