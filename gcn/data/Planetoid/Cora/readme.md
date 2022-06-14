# 数据集说明

主要包含以下文件：
+ ind.cora.x: 训练集节点特征向量，保存对象为: scipy.sparse.csr.csr_matrix，实际展开后大小为: (140, 1433)
+ ind.cora.tx: 测试集节点特征向量，保存对象为: scipy.sparse.csr.csr_matrix，实际展开后大小为: (1000, 1433)
+ ind.cora.allx: 包含有标签和无标签的训练节点特征向量，保存对象为: scipy.sparse.csr.csr_matrix, 实际展开后大小为: (1708, 1433)
+ ind.cora.y: one-hot表示的训练节点的标签，保存对象为: numpy.ndarray
+ ind.cora.ty: one-hot表示的测试节点的标签，保存对象为: numpy.ndarray
+ ind.cora.ally: ont-hot表示的ind.cora.allx对应的标签，保存对象为: numpy.ndarray
+ ind.cora.graph: 保存节点之间边的信息，保存格式为: {index: [index_of_neighbor_nodes]}
+ ind.cora.test.index: 保存测试节点的索引，保存对象为: List，用于后面的归纳学习设置