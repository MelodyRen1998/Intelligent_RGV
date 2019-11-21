# encoding: utf8

"""
数据结构: Priority Queue(优先队列)
算法: Depth First(深度优先搜索)

- `MinQueue`: 优先队列的实现。可以放入元素，使用 delMin 方法拿出 key 最小的一个元素。
- `RgvState`状态: 记录了某一时间点，RGV 机器在哪个位置，CNC 1-8 分别在做什么，有多少成品和半成品。
- `CNSState`状态: 记录了某台 CNC 机器的状态: 等待上料，加工中，等待下料
- `OperationTime`: RGV机器移动到某一个CNC机器的操作耗时

假定最有效率的情况下，成品数为 maxValue。
    (1) 从一个`RgvState`状态出发，列出所有可能的下一步操作，计算每一种可能性下的最大值 maxPossibleVaule，乘以-1作为 key，放入`MinQueue`中。
    (2) 从`MinQueue`中拿出 key 最小(可能成品数最大)的元素，继续列出所有可能的下一步操作，放入`MinQueue`中。

在计算下一步操作时：
    (a) 如果时间不足，则将当前操作步骤下的成品数量记录下来，与当前已有的 maxValue 对比，更新 maxValue
    (b) 如果 maxPossibleVaue < maxValue，剪枝，停止

**问题的关键**: 给定状态下, maxPossibleVaule 的计算。尽量压低 maxPossibleVaule，这样可以尽快剪枝停止。目前的算法不一定最优。
**可能的问题**:
    (1) RVG 的操作流程是否是: 1工序上料 -> 1工序下料 -> 2工序上料 -> 2工序清洗 -> 2工序下料。没搞太懂。
    (2) 传送带只能单项移动。那么在 CNC#7 上生产的半城品，是无法运到 CNC#2 上做第二道程序加工的。【这个肯定需要改】
    (3) 调换哪台 CNC 操作第几道工序，再重新运行？


Created on 2018.09.15

@author: admin
"""

from heapq import heappush, heappop
from copy import deepcopy


class OperationTime(object):
    def __init__(self, cnc: int=0, time: int=0, desc: list=None):
        """
        :param cnc: CNC机器编号
        :param time: 消耗时间
        :param desc: 描述
        """
        self.cnc = cnc
        self.time = time
        if not desc:
            self.desc = []

    def add(self, time=0, desc=''):
        self.time += time
        self.desc.append('%s(%s)' % (desc, time))

    def __str__(self):
        return '[%s]号CNC操作总耗时: %4s; 分步耗时: %s' %(self.cnc, self.time, self.desc)


class CNSState(object):
    def __init__(self, time=0, has_product=0):
        """
        机器状态
        :param time: 是否空闲。0：空闲; 其他: 表示开始加工的时间，与当前流逝时间对比
        :param has_product: 是否有成品

        (1) 等待上料:                        has_product = 0 & time = 0
        (2) 正在加工: 等待加工完成，做下料操作. has_product = 0 & time > 0
        (3) 加工完成: 直接下料                has_product = 1 & time = 0
        """
        self.has_product = has_product
        self.time = time

    def reset(self):
        self.has_product = 0
        self.time = 0

    def __str__(self):
        if self.has_product == 0:
            if self.time > 0:
                return '第 [%s]s 开始加工, 目前正在加工...' % self.time
            return '空闲状态'
        return '加工完成, 等待下料'


class RgvState(object):
    def __init__(self, maxValue=0, time=0, cnc_state_list=None, product=0, half_product=0, operations=None):
        """
        每个时间点: RGV的状态 和 cnc_state_list(各台机器的状态)
        :param maxValue: 剩余时间下能创造的最大价值
        :param time: 已流逝的时间
        :param cnc_state_list: 每台机器的状态. 消耗时长
        """
        self.maxValue = 0
        self.time = 0
        if not cnc_state_list:
            self.cns_state_list = [CNSState() for i in range(8)]
        self.product = 0
        self.half_product = 0
        self.move_list = [1]   # 移动路线
        if not operations:
            self.operations = []

    def __str__(self):
        return '成品数量: %s; 半成品数量: %s; 耗时: %s; 移动路径: %s' % (self.product, self.half_product, self.time, self.move_list)

    def __cmp__(self, other):
        # 用于 MinQueue 排序，数值小的排前边
        if self.product < other.product:
            return 1
        if self.product > other.product:
            return -1
        if self.half_product > other.product:
            return -1
        if self.half_product == other.product:
            return 0
        return 1

    def __lt__(self, other):
        return self.__cmp__(other) == -1

    def __gt__(self, other):
        return self.__cmp__(other) == 1

    def __eq__(self, other):
        return self.__cmp__(other) == 0


class MinQueue(object):
    def __init__(self):
        self.items = []

    def size(self):
        return len(self.items)

    def push(self, item):
        heappush(self.items, item)

    def popMin(self):
        return heappop(self.items)


class TwoStepRgvSolver(object):
    def __init__(self, move_speed=(20, 33, 46), process_time={1: 400, 2: 378},
                 wash_time=25, liao_time=[28, 31] * 4, max_time=8 * 3600,
                 order_dict=None):
        """

        :param move_time: RGV 移动1，2，3个单位所需的时间
        :param process: CNC加工完成一个两道工序物料的第一道工序所需时间 和 第二道工序所需时间
        :param wash_time: RGV完成一个物料的清洗作业所需时间
        :param liao_time: RGV为CNC1# - CNC8# 一次上下料所需时间
        :param max_time: 总时长，8小时
        :param order_dict: 每台机器负责第几道工序
        """
        self.move_speed = dict(zip(range(1, 4), move_speed))
        self.move_speed[0] = 0
        self.process_time = process_time
        self.wash_time = wash_time
        self.liao_time = liao_time
        self.max_time = max_time
        if not order_dict:
            self.order_dict = {1: 1, 2: 2, 3: 1, 4: 2,
                               5: 1, 6: 2, 7: 1, 8: 2}
        self.possible_cns = list(range(1, 9))
        self.__set_dist()

        self.RgvStateQueue = MinQueue()
        self.optimizeState = RgvState()

        # 1上料 + 1下料 + 2上料 + 清洗 + 2下料
        self.min_product_time = min(self.liao_time) * 2 + max(self.liao_time) * 2 +\
                                self.wash_time + \
                                sum(self.process_time.values())

    def __set_dist(self):
        # 每台机器之间的距离
        dist = {(1, 1): 0, (2, 2): 0, (3, 3): 0, (4, 4): 0,
                (5, 5): 0, (6, 6): 0, (7, 7): 0, (8, 8): 0,
                (1, 2): 0, (1, 3): 1, (1, 4): 1, (1, 5): 2, (1, 6): 2, (1, 7): 3, (1, 8): 3,
                (2, 3): 1, (2, 4): 1, (2, 5): 2, (2, 6): 2, (2, 7): 3, (2, 8): 3,
                (3, 4): 0, (3, 5): 1, (3, 6): 1, (3, 7): 2, (3, 8): 2,
                (4, 5): 1, (4, 6): 1, (4, 7): 2, (4, 8): 2,
                (5, 6): 0, (5, 7): 1, (5, 8): 1,
                (6, 7): 1, (6, 8): 1,
                (7, 8): 0
                }
        more_dist = {(s2, s1): self.move_speed[d] for (s1, s2), d in dist.items() if s1 < s2}
        for (s1, s2), d in dist.items():
            more_dist[(s1, s2)] = self.move_speed[d]
        self.move_time = more_dist

    def _move_cost_time(self, state_from: int, state_to: int):
        # 从一台机器移动到另一台机器的时间
        return self.move_time[(state_from, state_to)]

    def is_timeout(self, state: RgvState):
        # 检查是否超时 & 如果超时，则检测最终的状态是否是最优的
        if state.time > self.max_time:
            # print('timeout state: ', state.time, state)
            self.update_optimize_state(state)
            return True
        return False

    def update_optimize_state(self, state: RgvState):
        # 与最优状态对比，先对比成品数量，相等的话再对比成品数量
        if state.product < self.optimizeState.product:
            return
        if state.product > self.optimizeState.product:
            self.optimizeState = state
        elif state.half_product > self.optimizeState.product:
            self.optimizeState = state

    def move(self, state: RgvState, cnc_to: int):
        # 从当前状态移动到第 cnc_to 台机器进行操作
        # print('move state: ', state)
        operation = OperationTime(cnc_to)
        cnc_from = state.move_list[-1]
        time_cost = self._move_cost_time(cnc_from, cnc_to)
        state.time += time_cost
        # 移动超时
        if self.is_timeout(state):
            return
        operation.add(time_cost, '移动时长')
        res = self.update_state_cnc(state, cnc_to, operation)
        # 状态更新超时
        if not res:
            return
        state, operation = res
        state.operations.append(operation)
        # 更新移动路径 & 修改最佳剩余时间
        state.move_list.append(cnc_to)
        # print('更新状态  %s -> %s; 移动耗时: %s; 总耗时: %s; move: %s' % (cnc_from, cnc_to, time_cost, state.time, state.move_list))
        state.maxValue = self.get_max_value(state)
        self.update_optimize_state(state)
        return state

    def update_state_cnc(self, state: RgvState, cnc_order: int, operation:OperationTime):
        """
        到达某个 CNC 机器后，对 CNC 机器做操作。机器可能的状态:
        (1) 等待上料:                        has_product = 0 & time = 0
        (2) 正在加工: 等待加工完成，做下料操作. has_product = 0 & time > 0·
        (3) 加工完成: 直接下料                has_product = 1 & time = 0
        更新总体的状态
        :return:
        """
        cnc_state = state.cns_state_list[cnc_order - 1]
        liao_time = self.liao_time[cnc_order - 1]   # 上料 or 下料时间
        product_order = self.order_dict[cnc_order]  # 1: 加工第一道工序，2: 加工第二道工序
        # print('\n%s 号 cnc 加工第[%s]道工序； 状态: %s;' % (cnc_order, product_order, cnc_state))

        # 等待加工
        if cnc_state.time > 0:
            wait_time = cnc_state.time + self.process_time[product_order] - state.time
            if wait_time > 0:  # 加工未完成, 继续等待一会儿
                state.time += wait_time
                if self.is_timeout(state):
                    return
                operation.add(wait_time, '等待加工')
            # 更新 has_product 状态
            cnc_state.has_product = 1
            cnc_state.time = 0


        # 加工完成，等待下料 or 洗料
        if cnc_state.has_product:
            if product_order == 1:
                # 增加下料消耗的时间 & 检测超时
                state.time += liao_time
                if self.is_timeout(state):
                    return
                operation.add(liao_time, '一工序下料')
                # 完成第一道工序，半成品 +1
                state.half_product += 1
                cnc_state.has_product = 0
                cnc_state.time = state.time
            else:
                # 完成第二道工序  -> 洗料 -> 下料
                state.time += liao_time
                if self.is_timeout(state):
                    return
                operation.add(liao_time, '二工序下料')
                state.time += self.wash_time
                if self.is_timeout(state):
                    return
                operation.add(self.wash_time, '二工序洗料')
                state.product += 1
                if state.half_product > 0:
                    cnc_state.has_product = 0
                    cnc_state.time = state.time
                    return
                else:
                    cnc_state.reset()

            # 重置状态
            state.cns_state_list[cnc_order - 1] = cnc_state
            return state, operation


        # 等待上料
        # 给第一道工序加料
        if product_order == 1:
            # 增加上料消耗的时间 & 检测超时
            state.time += self.liao_time[cnc_order - 1]
            if self.is_timeout(state):
                return
            operation.add(liao_time, '一工序上料')
            cnc_state.time = state.time
            state.cns_state_list[cnc_order - 1] = cnc_state
        # 将半成品拿给第二道工序加料
        elif state.half_product > 0:
            # 增加上料消耗的时间 & 检测超时
            state.time += liao_time
            if self.is_timeout(state):
                return
            operation.add(liao_time, '二工序上料')
            state.half_product -= 1
            cnc_state.time = state.time
            state.cns_state_list[cnc_order - 1] = cnc_state
        # 无半成品，只好什么都不做，这样移动时间就白白消耗了，不返回该状态
        else:
            return
        return state, operation

    def get_max_value(self, state: RgvState):
        """
        给定状态，计算可能的最优目标函数
        (1) 已成功完成两道工序 而且已经清洗下料的成品数量
        (2) 未来得及下料数量
        (3) 除去下料时间 + 剩余时间可能的成品数量
        :param state:
        :return:
        """
        # 随意吧，把半成品都当做成品好了，不可放过
        full_pro_num = state.product
        gone_time = state.time
        for cnc in state.cns_state_list:
            if cnc.has_product:
                full_pro_num += 1
                gone_time += min(self.liao_time)
        full_pro_num += (self.max_time - gone_time) / self.min_product_time
        return full_pro_num

    def depth_first_search(self):
        # 行吧，开始搜吧
        cout = 0
        while self.RgvStateQueue.size() > 0:
            val, state = self.RgvStateQueue.popMin()
            # print('val: %s; pop state: %s' % (val, state))
            # 剪枝，结束
            if state.maxValue < self.optimizeState.product:
                print('剪枝，结束')
                break
            # 继续向深处搜索吧少年
            for cnc in self.possible_cns:
                s = deepcopy(state)
                s = self.move(s, cnc)
                if s and s.maxValue > self.optimizeState.product:
                    self.RgvStateQueue.push((-s.maxValue, s))
            cout += 1
            # print('length of RgvStateQueue: ', self.RgvStateQueue.size())
        # while self.RgvStateQueue.size():
        #     k, v = self.RgvStateQueue.popMin()
        #     print(k, v)

    def main(self):
        # 这个`最多产生成品`的算法是不对的，因为加工时可以做上下料的操作，不必一直等待
        print('最多产生成品: %.2f' % (self.max_time / self.min_product_time))

        self.RgvStateQueue.push((0, RgvState()))
        self.depth_first_search()
        print('最优状态: ')
        print(self.optimizeState)
        print('操作步骤: ')
        for o in self.optimizeState.operations:
            print(o)


if __name__ == '__main__':
    # queue = MinQueue()
    # queue.push((1, 3))
    # queue.push((2, 3))
    # queue.push((10, 3))
    # print(queue.popMin())
    # print(queue.popMin())
    # a, b = queue.popMin()
    # print(a, '->', b)

    # from RNG import *
    solver = TwoStepRgvSolver()
    import time as timer
    t1 = timer.time()
    solver.main()
    print('time cost: ', timer.time() - t1)
