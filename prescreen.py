# Anthropic prescreen prep
# Looks like the format is you implement a class to do something
# And you have a function that gets called on a list of commands
# You iterate throught the list of commands and call the class function on each command

class IntContainer:
    def __init__(self):
        self.iList = []
        
    def add(self, value):
        self.iList.append(value)
        return ""
            
    def exists(self, value):
        if value in self.iList:
            return "true"
        else:
            return "false"

    def remove(self, value):
        if value in self.iList:
            self.iList.remove(value)
            return "true"
        else:
            return "false"

    def get_next(self, value):
        ret_string = ""
        ret_value = value
        for elem in self.iList:
            if elem > value: # could be closest
                if ret_value == value:
                    ret_value = elem
                elif elem < ret_value:
                    ret_value = elem
                    
        if ret_value != value:
            ret_string = str(ret_value)
            
        return ret_string
        
def solution(queries):
    container = IntContainer()
    ret_list = []
    
    for query in queries:
        if query[0] == "ADD":
            ret_list.append(container.add(int(query[1])))
            continue
        if query[0] == "EXISTS":
            ret_list.append(container.exists(int(query[1])))
            continue
        if query[0] == "REMOVE":
            ret_list.append(container.remove(int(query[1])))
            continue
        if query[0] == "GET_NEXT":
            ret_list.append(container.get_next(int(query[1])))
            continue
            
    return ret_list

queries = [
    ["ADD", "1"],
    ["ADD", "2"],
    ["ADD", "2"],
    ["ADD", "4"],
    ["GET_NEXT", "1"],
    ["GET_NEXT", "2"],
    ["GET_NEXT", "3"],
    ["GET_NEXT", "4"],
    ["REMOVE", "2"],
    ["GET_NEXT", "1"],
    ["GET_NEXT", "2"],
    ["GET_NEXT", "3"],
    ["GET_NEXT", "4"]
]
results = solution(queries)
assert results == ["", "", "", "", "2", "4", "4", "", "true", "2", "4", "4", ""]
print("All tests pass")
