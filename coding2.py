
"""
Assumption 1: 
Low priority is given to accessing segments by index.
High priority is given to adding or modifying segments. 
-----> use linked list to allow O(1) new segment adding/merging operation.

Assumption 2:
Support [-inf, x], [x, inf], [-inf, inf]. 
"""


class segment:
    def __init__(self, start, intensity, nxt = None):
        self.next = nxt
        self.start = start
        self.intensity = intensity


class Intensity:
    
    def __init__(self):
        self.head = segment(float('-inf'), 0)
        self.tail = segment(float('inf'), 0)
        self.head.next = self.tail
    
    @staticmethod
    def check(left, right, amount):
        assert isinstance(amount,int ), "Require amount is integer"
        assert left < float('inf'), "Require left < inf"
        assert right > float('-inf'), "Require right > - inf"
        assert left < right, "Require left < right"

    def get(self):
        ans = []
        cur = self.head
        while cur:
            ans.append([cur.start, cur.intensity])
            cur = cur.next
        ans.pop()
        if len(ans) == 1 and ans[0] == 0:
            return []
        return ans[1:] if ans[0][1] == 0 else ans

    def add(self, left, right, amount):
        Intensity.check(left, right, amount)

        if amount == 0:
            return
        
        left_create = True
        prev = self.head.intensity
        if left == float('-inf'):
            self.head.intensity += amount

        p1 = self.head
        p2 = p1.next
        
        while p2:
            if p1.start < left and p2.start > left and left_create:
                prev = p1.intensity
                tmp = segment(start = left, intensity = p1.intensity + amount, nxt = p2)
                p1.next = tmp
                p1 = p1.next
            if p2.start >= right: 
                break
            if p2.start == left:
                left_create = False
                prev = p2.intensity
                p2.intensity += amount
                if p2.intensity == p1.intensity:
                    p1.next = p2.next
                    p2 = p2.next
                    continue
            if left < p2.start < right:
                prev = p2.intensity
                p2.intensity += amount
            
            p1 = p1.next
            p2 = p2.next

        if p2.start == right:
            if p1.intensity == p2.intensity:
                p1.next = p2.next or self.tail
            
        if p2.start > right:
            tmp = segment(start = right, intensity = prev, nxt = p2)
            p1.next = tmp

        

    def set(self, left , right, amount):
        Intensity.check(left, right, amount)

        prev = self.head.intensity
        if left == float('-inf'):
            self.head.intensity = amount

        left_create = True

        p1 = self.head
        p2 = p1.next
        
        while p2:
            if p1.start < left and p2.start > left and left_create:
                prev = p1.intensity
                if p1.intensity != amount:
                    tmp = segment(start = left, intensity = amount, nxt = p2)
                    p1.next = tmp
                    p1 = p1.next
            if p2.start >= right: 
                break
            if p2.start == left:
                left_create = False
                prev = p2.intensity
                p2.intensity = amount
                if p2.intensity == p1.intensity:
                    p1.next = p2.next
                    p2 = p2.next
                    continue
            if left < p2.start < right:
                prev = p2.intensity
                p2 = p2.next
                continue
            
            p1 = p1.next
            p2 = p2.next
    
                    
        if p2.start > right and prev!= amount:
            tmp = segment(start = right, intensity = prev, nxt = p2)
            p1.next = tmp
        
        else:
            p1.next = p2
            if p1.intensity == p2.intensity:
                p1.next = p2.next or self.tail

        






# a.add(10, 30, 1)
# print(a.get())
# a.add(20, 40, 1)
# print(a.get())
# a.add(10, 40.5, -1)
# print(a.get())
# a.add(10, 40.5, -1)
# print(a.get())


# a.add(1, 10, -2)
# print(a.get())
# a.add(1, 10, 2)
# print(a.get())


# a.set(1, 10, -2)
# print(a.get())
# a.set(float('-inf'), 5, 2)
# print(a.get())