# -*- coding: utf-8 -*-
# @Author: Wei Li
# @Date:   2019-04-14 15:49:39
# @Last Modified by:   liwei
# @Last Modified time: 2019-05-05 22:44:14

from py2neo import Graph, Node, Relationship
import json


# 读取三元组
def readTriples(session_file):
    knowledges = []
    goals = []
    with open(session_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            session = json.loads(line.strip())
            knowledges.append(session['knowledge'])
            goals.append(session['goal'])
    return knowledges, goals


# 图中增加一个三元组
def addTriple(graph, a, relation, b, propA='topics', propB='props'):
    # 图中查找节点 a, b，如果没有则创建
    if not len(graph.run("MATCH(p:%s{name:'%s'}) return p" % (propA, a)).data()):
        graph.create(Node(propA, name=a))
    if not len(graph.run("MATCH(q:%s{name:'%s'}) return q" % (propB, b)).data()):
        graph.create(Node(propB, name=b))
    # 创建 (a, r, b)
    nodeA = graph.run("MATCH(p:%s{name:'%s'}) return p" %
                      (propA, a)).data()[0]['p']
    nodeB = graph.run("MATCH(q:%s{name:'%s'}) return q" %
                      (propB, b)).data()[0]['q']
    graph.create(Relationship(nodeA, relation, nodeB))


# 创建一个会话 knowledge 的知识图
def addOneSessGraph(graph, knowledges, goal):
    topicA, topicB = goal[0][1], goal[0][2]
    topicRelation = goal[1][1]
    addTriple(graph, topicA, topicRelation, topicB,
              propA='topics', propB='topics')

    for a, r, b in knowledges:
        addTriple(graph, a, r, b)


knowledges, goals = readTriples('../data/resource/dev.txt')
print(knowledges[0])
print(goals[0])

# 连接neo4j数据库，输入地址、用户名、密码
graph = Graph('http://localhost:7474', username='neo4j', password='123456')
graph.delete_all()

i = 0
for knowledge, goal in zip(knowledges, goals):
    addOneSessGraph(graph, knowledge, goal)
    i += 1
    if i > 5:
        break


# 读取对话
def readDialogues(session_file):
    sessions = []
    with open(session_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            session = json.loads(line.strip())
            sessions.append(session['conversation'])
    return sessions


def writeDialogues(file, sessions):
    with open(file, 'w', encoding='utf-8') as f:
        for session in sessions:
            f.write('\n'.join(session))
            f.write('\n\n')


# writeDialogues('../data/resource/sess.dev.txt',
#                readDialogues('../data/resource/dev.txt'))
