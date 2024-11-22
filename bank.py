from cloud_storage import CloudStorage


import re

class CloudStorageImpl(CloudStorage):

    def __init__(self):
        self.name_to_info = {}
        self.user_to_capacity = {}
            
    def add_file(self, name: str, size: int, user_id="admin") -> bool:
        """
        Should add a new file `name` to the storage.
        `size` is the amount of memory required in bytes.
        The current operation fails if a file with the same `name`
        already exists.
        Returns `True` if the file was added successfully or `False`
        otherwise.
        """
        # default implementation
        if name in self.name_to_info:
            return False
        else:
            self.name_to_info[name] = [size,user_id]
            return True

    def copy_file(self, name_from: str, name_to: str) -> bool:
        """
        Should copy the file at `name_from` to `name_to`.
        The operation fails if `name_from` points to a file that
        does not exist or points to a directory.
        The operation fails if the specified file already exists at
        `name_to`.
        Returns `True` if the file was copied successfully or
        `False` otherwise.
        """
        # default implementation
        if name_to in self.name_to_info:
            return False
        
        if name_from in self.name_to_info :      
            user_id = self.name_to_info[name_from][1]
            if user_id != "admin" and (self.name_to_info[user_id][0] - self.name_to_info[user_id][1]) > size:
                return False
            size = self.name_to_info[name_from][0]
            self.name_to_info[user_id][1] += size
            
            # 
            print(size, user_id)
            # ret = self.add_file_by(user_id, name_to, size)
            # if ret != None: 
            #     return True
            # else:
            #     return False
            self.name_to_info[name_to] = self.name_to_info[name_from]
            return True
            # del self.name_to_info[name_from]
        
        return False

    def get_file_size(self, name: str) -> int | None:
        """
        Should return the size of the file `name` if it exists, or
        `None` otherwise.
        """
        # default implementation
        if name in self.name_to_info:
            return self.name_to_info[name][0]
            
        return None
            
    def find_file(self, prefix: str, suffix: str) -> list[str]:
        """
        Should search for files with names starting with `prefix`
        and ending with `suffix`.
        Returns a list of strings representing all matching files in
        this format:
        `["<name_1>(<size_1>)", "<name_2>(<size_2>)", ...]`.
        The output should be sorted in descending order of file
        sizes or, in the case of ties,
        [lexicographically](keyword://lexicographical-order-for-
        strings).
        If no files match the required properties, should return an
        empty list.
        """
        # default implementation
        
        pattern = re.compile("^" + prefix + ".*" + suffix + "$")
        ans = []
        for k,value in self.name_to_info.items():
            if pattern.match(k):
                v = value[0]
                ans.append((-v, k, k+"("+str(v)+")"))
        
        ans.sort()
        ans1 = [x[2] for x in ans]
        return ans1
    
    def add_user(self, user_id: str, capacity: int) -> bool:
        """
        Should add a new user to the system, with `capacity` as
        their storage limit in bytes.
        The total size of all files owned by `user_id` cannot exceed
        `capacity`.
        The operation fails if a user with `user_id` already exists.
        Returns `True` if a user with `user_id` is successfully
        created, or `False` otherwise.
        """
        # default implementation
        if user_id in self.user_to_capacity:
            return False
            
        self.user_to_capacity[user_id] = [capacity, 0, []] # total capacity, used capacity, list of files owned by user
        return True

    def add_file_by(self, user_id: str, name: str, size: int) -> int | None:
        """
        Should behave in the same way as the `add_file` from Level
        1, but the added file should be owned by the user with
        `user_id`.
        A new file cannot be added to the storage if doing so will
        exceed the user's `capacity` limit.
        Returns the remaining storage capacity for `user_id` if the
        file is successfully added or `None` otherwise.
        
        *Note that* all queries calling the `add_file` operation
        implemented during Level 1 are run by the user with
        `user_id = "admin"`, who has unlimited storage capacity.
        Also, assume that the `copy_file` operation preserves the
        ownership of the original file.
        """
        # default implementation
        
        if user_id not in self.user_to_capacity and user_id != "admin"
            return None
        
        if size > (self.user_to_capacity[user_id][0] - self.user_to_capacity[user_id][1]) and user_id != "admin":
            return None
            
        if self.add_file(name, size, user_id):            
            if user_id != "admin":
                self.user_to_capacity[user_id][1] += size # reduce capacity left for user
            return (self.user_to_capacity[user_id][0] - self.user_to_capacity[user_id][1])
        else:
            return None

    def update_capacity(self, user_id: str, capacity: int) -> int | None:
        """
        Should change the maximum storage capacity for the user with
        `user_id`.
        If the total size of all user's files exceeds the new
        `capacity`, the largest files (sorted
        [lexicographically](keyword://lexicographical-order-for-
        strings) in case of a tie) should be removed from the
        storage until the total size of all remaining files will no
        longer exceed the new `capacity`.
        Returns the number of removed files, or `None` if a user
        with `user_id` does not exist.
        """
        # default implementation
        if user_id not in self.user_to_capacity:                
            return None
            
        num_removed = 0
        if self.user_to_capacity[user_id][1] > capacity: # user has too many files
            # !!! remove files in order of size till less than capacity
            num_removed += 1
            
        self.user_to_capacity[user_id][1] = capacity
        return num_removed



# do simple time conversion - string to time object - and print the time object

import time

string = "Tue, 03 Aug 2021 10:45:08"
obj = time.strptime(string, "%a, %d %b %Y %H:%M:%S")
print(obj)
print(obj.tm_year)
string = "03 Aug 2021 10:45:08"
obj = time.strptime(string, "%d %b %Y %H:%M:%S")
print(obj)
print(obj.tm_year)
string = "03 Aug 2021 17:45:08"
obj1 = time.strptime(string, "%d %b %Y %H:%M:%S")
print(obj1)

import time
obj = time.gmtime(1627987508.6496193)
time_str = time.asctime(obj)
print(time_str)
obj = time.localtime(1627987508.6496193)
time_str = time.asctime(obj)
print(time_str)

from datetime import datetime

def getDuration(then, now = datetime.now(), interval = "default"):

    # Returns a duration as specified by variable interval
    # Functions, except totalDuration, returns [quotient, remainder]

    duration = now - then # For build-in functions
    duration_in_s = duration.total_seconds() 
    
    def years():
      return divmod(duration_in_s, 31536000) # Seconds in a year=31536000.

    def days(seconds = None):
      return divmod(seconds if seconds != None else duration_in_s, 86400) # Seconds in a day = 86400

    def hours(seconds = None):
      return divmod(seconds if seconds != None else duration_in_s, 3600) # Seconds in an hour = 3600

    def minutes(seconds = None):
      return divmod(seconds if seconds != None else duration_in_s, 60) # Seconds in a minute = 60

    def seconds(seconds = None):
      if seconds != None:
        return divmod(seconds, 1)   
      return duration_in_s

    def totalDuration():
        y = years()
        d = days(y[1]) # Use remainder to calculate next variable
        h = hours(d[1])
        m = minutes(h[1])
        s = seconds(m[1])

        return "Time between dates: {} years, {} days, {} hours, {} minutes and {} seconds".format(int(y[0]), int(d[0]), int(h[0]), int(m[0]), int(s[0]))

    return {
        'years': int(years()[0]),
        'days': int(days()[0]),
        'hours': int(hours()[0]),
        'minutes': int(minutes()[0]),
        'seconds': int(seconds()),
        'default': totalDuration()
    }[interval]

# Example usage
then = datetime(2020, 3, 5, 23, 8, 15)
now = datetime.now()

print(getDuration(then)) # E.g. Time between dates: 7 years, 208 days, 21 hours, 19 minutes and 15 seconds
print(getDuration(then, now, 'years'))      # Prints duration in years
print(getDuration(then, now, 'days'))       #                    days
print(getDuration(then, now, 'hours'))      #                    hours
print(getDuration(then, now, 'minutes'))    #                    minutes
print(getDuration(then, now, 'seconds'))    #                    seconds


# Search a list of strings for a specific string with a regular expression matcher

import re
word_list = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "Parker", "Patrick", "peter", "piper", "pickled", "peppers"]

def search_list(word_list, search_string):
    pattern = re.compile(search_string, re.IGNORECASE)
    return [word for word in word_list if pattern.match(word)]

search_string = "p.*"
print(search_list(word_list, search_string))
search_string = "[pP].*"
print(search_list(word_list, search_string))
search_string = "[pP].*"
print(search_list(word_list, search_string))
search_string = "p.*r"
print(search_list(word_list, search_string))
search_string = "p.*r.*"
print(search_list(word_list, search_string))
search_string = "p.*r.*k"
print(search_list(word_list, search_string))
search_string = "p.*r.*k.*"
print(search_list(word_list, search_string))




exit()
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

def solution(queries, answers):
    container = IntContainer()
    ret_list = []
    i_query = 0
    
    for query, answer in zip(queries, answers):
        if query[0] == "ADD":
            ret_list.append(container.add(int(query[1])))
            continue
        elif query[0] == "EXISTS":
            ret_list.append(container.exists(int(query[1])))
            continue
        elif query[0] == "REMOVE":
            ret_list.append(container.remove(int(query[1])))
            continue
        elif query[0] == "GET_NEXT":
            ret_list.append(container.get_next(int(query[1])))
            continue
        else:
            print(f"Invalid {i_query=} {query=}")
            return False

        if ret_list[-1] != answer:
            print(f"{i_query=} {ret_list[-1]=} {answer=} {query=}")
            return False

        i_query += 1
            
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
answers = ["", "", "", "", "2", "4", "4", "", "true", "2", "4", "4", ""]
results = solution(queries, answers)
assert results == answers
print("All IntContainer tests pass")

class Account:
    def __init__(self, account_number, customer, initial_balance=0, bank=None):
        self.account_number = account_number
        self.customer = customer
        self.balance = initial_balance
        self.transactions = []
        self.status = "active" # active, merged, closed
        self.merged_into = None # account number of the account this account was merged into
        self.bank = bank

    def __str__(self):
        return f"Account[{self.account_number}] - Holder: {self.account_holder} - Balance: ${self.balance}"

    def deposit(self, amount, from_account=None):
        if self.status != "active":
            print("Account is not active.")
            return None
        if amount > 0:
            self.balance += amount
            # self.transactions.append(f"Deposited: ${amount} from ${from_account}")
            self.log_transaction("deposit", amount, from_account=from_account, to_account=self.account_number)
            print(f"${amount} deposited successfully.")
            return self.balance
        else:
            print("Deposit amount must be positive.")
        return None

    def withdraw(self, amount, to_account=None, force=False):
        if self.status != "active":
            print("Account is not active.")
            return None
        if amount > 0:
            if amount <= self.balance or force:
                self.balance -= amount
                # self.transactions.append(f"Withdrew: ${amount} to ${to_account}")
                self.log_transaction("withdraw", amount, from_account=self.account_number, to_account=to_account)
                print(f"Account withdraw {amount} balance {self.balance} withdrawn successfully.")
                return self.balance
            else:
                print("Insufficient funds.")
        else:
            print("Withdrawal amount must be positive.")
        return None

    def check_balance(self):
        if self.status != "active":
            print("Account is not active.")
            return None
        print(f"Current balance: ${self.balance}")
        return self.balance

    def log_transaction(self, transaction_type, amount, from_account=None, to_account=None):
        self.transactions.append({
            'type': transaction_type,  # withdraw, deposit, transfer
            'from': from_account,
            'to': to_account,
            'amount': amount,
            'balance': self.balance
        })

    def view_transactions(self):
        print("Transaction History:")
        for transaction in self.transactions:
            print(transaction)

class Bank:
    def __init__(self):
        self.accounts = {}
        self.next_account_number = 1
        self.customer_to_account = {}
        self.next_transaction_id = 1
        self.sched_trans = []

    def CreateAccount(self, customer, account_number, initial_balance):
        if account_number in self.accounts:
            print("Account number already exists.")
            return None

        if account_number == -1:
            account_number = self.next_account_number
            self.next_account_number += 1
        print(f"Account number: {account_number}")

        self.accounts[account_number] = Account(account_number, customer, initial_balance, bank=self)
        account_list = self.customer_to_account.get(customer, [])
        account_list.append(account_number)
        print(f"Account list: {account_list}")
        self.customer_to_account[customer] = account_list
        return account_number

    def get_account(self, account_number):
        return self.accounts.get(account_number)

    def Deposit(self, account_number, amount):
        account = self.get_account(account_number)
        if account:
            return account.deposit(amount)
        return None

    def Withdraw(self, account_number, amount):
        account = self.get_account(account_number)
        if account:
            return account.withdraw(amount)
        return None

    def GetBalance(self, account_number):
        account = self.get_account(account_number)
        if account:
            return account.check_balance()
        return None

    def Transfer(self, from_account_number, to_account_number, amount, force=False):
        from_account = self.get_account(from_account_number)
        to_account = self.get_account(to_account_number)
        if from_account and to_account:
            amount_left = from_account.withdraw(amount, to_account_number, force)
            print(f"Transfer: Amount left after withdraw: {amount_left}")
            if amount_left != None and (amount_left >= 0 or force): # force allows a negative balance to happen
                to_account.deposit(amount, from_account_number)
                return amount_left
            else:
                print("Transfer failed")
        return None

    def GetCustomerAccounts(self, customer):
        account_list = self.customer_to_account.get(customer,None)
        return account_list

    def GetAccountTransactions(self, account_number):
        account = self.get_account(account_number)
        if account:
            account.view_transactions()
            return account.transactions
        return None

    def GetTopKAccounts(self, k):
        top_accounts = [] # list of tuples (score, account_number)

        for account_number, account in self.accounts.items():
            # compute the total sum of transactions in the account
            sum_transactions = 0
            for transaction in account.transactions:
                sum_transactions += abs(transaction['amount'])

            if len(top_accounts) < k:
                top_accounts.append((sum_transactions, account_number))
                top_accounts.sort(reverse=True)
            elif top_accounts[-1][0] < sum_transactions:
                top_accounts.append((sum_transactions, account_number))
                top_accounts.sort(reverse=True)
                top_accounts.pop()
        return [account_number for score, account_number in top_accounts]

    def ScheduleTransfer(self, account_from, account_to, amount):
        self.sched_trans.append((account_from, account_to, amount))
        return True

    def CancelTransfer(self, account_from, account_to, amount):
        if (account_from, account_to, amount) in self.sched_trans:
                self.sched_trans.remove((account_from, account_to, amount))
                return True

        return False

    def ExecuteScheduledTransfers(self):
        # First do them all and force negative balance to succeed
        failed = []
        for account_from, account_to, amount in self.sched_trans:
            amount_left = self.Transfer(account_from, account_to, amount, True)
            print(f"Excecute Transfer: {account_from} -> {account_to} - {amount} - {amount_left}")
            if amount_left == None: # or amount_left < 0:
                print(f"Excecute Transfer failed: {account_from} -> {account_to} - {amount}")
                failed.append((account_from, account_to, amount))

        # Then undo all the failed ones by reversing the transfers that resulted in negative balance
        bCheck = True
        while bCheck:
            bCheck = False
        for account in self.accounts.values():
            print(f"Execute Account {account.account_number} has balance: {account.balance}")
            if account.balance < 0:
                print(f"Account {account.account_number} has negative balance: {account.balance}")
                bCheck = True # have to check again - as we may push other accounts balance negative
                # reverse the transactions till it's positive
                reversed_trans = reversed(list(enumerate(account.transactions)))
                for idx, transaction in reversed_trans:
                    if transaction['from'] == account.account_number:
                        print("reverse the transactions till it's positive")
                        self.Transfer(transaction['to'], transaction['from'], transaction["amount"], True)
                        account.balance += transaction['amount']
                        failed.append((transaction['from'], transaction['to'], transaction['amount']))
                        print("failed.append((transaction['from'], transaction['to'], transaction['amount']))")
                        del account.transactions[idx]
                        del account.transactions[-1]
                        if account.balance >= 0:
                            break

        return failed

    def MergeAccounts(self, account_number_trg, account_number_src):
        if account_number_trg not in self.accounts:
            return None
        if account_number_src not in self.accounts:
            return None

        account_trg = self.accounts[account_number_trg]
        account_src = self.accounts[account_number_src]

        # def log_transaction(self, transaction_type, amount, from_account=None, to_account=None):

        account_trg.balance += account_src.balance

        account_src.status = "merged"
        account_src.balance = 0
        account_src.merged_into = account_number_trg
        account_trg.log_transaction("merge", account_src.balance, from_account=account_number_src, to_account=account_number_trg)
        account_src.log_transaction("merge", account_src.balance, from_account=account_number_src, to_account=account_number_trg)
        return True

# last parameter is the expected return value
queries = [
    ["CreateAccount", "A", -1, 0.0, 1],
    ["CreateAccount", "B", -1, 0.0, 2],
    ["CreateAccount", "A", -1, 0.0, 3],
    ["CreateAccount", "B", 4, 0.0, 4],
    ["CreateAccount", "c", 1, 0.0, None], # duplicate account number
    ["Deposit", 1, 100.0, 100.0],
    ["Deposit", 2, 200.0, 200.0],
    ["Deposit", 3, 300.0, 300.0],
    ["Deposit", 4, 400.0, 400.0],
    ["Deposit", 5, 500.0, None], # account does not exist
    ["GetBalance", 1, 100.0],
    ["GetBalance", 2, 200.0],
    ["GetBalance", 3, 300.0],
    ["GetBalance", 4, 400.0],
    ["GetBalance", 5, None],
    ["Withdraw", 1, 50.0, 50.0],
    ["Withdraw", 2, 50.0, 150.0],
    ["Withdraw", 3, 50.0, 250.0],
    ["Withdraw", 4, 50.0, 350.0],
    ["Withdraw", 5, 50.0, None], # account does not exist
    ["GetBalance", 1, 50.0],
    ["GetBalance", 2, 150.0],
    ["GetBalance", 3, 250.0],
    ["GetBalance", 4, 350.0],
    ["GetBalance", 5, None],
    ["Transfer", 1, 2, 25.0, 25.0],
    ["Transfer", 2, 3, 25.0, 150.0],
    ["Transfer", 3, 4, 25.0, 250.0],
    ["Withdraw", 4, 1000.0, None], # insufficient funds
    ["Transfer", 4, 1, 1000.0, None], # insufficient funds
    ["Transfer", 4, 5, 25.0, None], # account does not exist
    ["Transfer", 5, 1, 25.0, None], # account does not exist
    ["GetBalance", 1, 25.0],
    ["GetBalance", 2, 150.0],
    ["GetBalance", 3, 250.0],
    ["GetBalance", 4, 375.0],
    ["GetBalance", 5, None],
    ["GetCustomerAccounts", "A", [1, 3]],
    ["GetCustomerAccounts", "B", [2, 4]],
    ["GetTopKAccounts", 4, [4, 3, 2, 1]],
    ["GetTopKAccounts", 7, [4, 3, 2, 1]],
    ["GetTopKAccounts", 2, [4, 3]],
    ["GetAccountTransactions", 1, [{'type': 'deposit', 'from': None, 'to': 1, 'amount': 100.0, 'balance': 100.0},
                                   {'type': 'withdraw', 'from': 1, 'to': None, 'amount': 50.0, 'balance': 50.0},
                                   {'type': 'withdraw', 'from': 1, 'to': 2, 'amount': 25.0, 'balance': 25.0}]],
    ["GetAccountTransactions", 5, None],
    ["ScheduleTransfer", 1, 2, 100.0, True],
    ["ScheduleTransfer", 2, 1, 100.0, True],
    ["ScheduleTransfer", 2, 6, 100.0, True],
    ["ExecuteScheduledTransfers", [(2, 6, 100.0)]],  # 47 index in test script
    ["GetBalance", 1, 25.0],
    ["GetBalance", 2, 150.0], # 49 index in test script
    ["GetBalance", 3, 250.0],
    ["GetBalance", 4, 375.0],
    ["GetBalance", 5, None],
    ["CancelTransfer", 4, 3, 400.0, False],
    ["CancelTransfer", 1, 2, 100.0, True],
    ["CancelTransfer", 2, 1, 100.0, True],
    ["CancelTransfer", 2, 6, 100.0, True],
    ["ExecuteScheduledTransfers", []],
    # Test a loop that pushes everyone negative works if they go positive again
    ["ScheduleTransfer", 1, 2, 1000.0, True],
    ["ScheduleTransfer", 3, 4, 1000.0, True],
    ["ScheduleTransfer", 2, 3, 1000.0, True],
    ["ScheduleTransfer", 4, 1, 1000.0, True],
    ["ExecuteScheduledTransfers", []], # 62
    # Test a loop that pushes everyone negative doesn't work if they don't go positive again
    ["ScheduleTransfer", 1, 2, 1000.0, True],
    ["ScheduleTransfer", 3, 4, 1000.0, True],
    ["ScheduleTransfer", 2, 3, 1000.0, True],
    ["ExecuteScheduledTransfers", []],

    ["MergeAccounts", 2, 1, True],
    ["GetBalance", 2, 175.0], 
    ["GetAccountTransactions", 1, [{'type': 'deposit', 'from': None, 'to': 1, 'amount': 100.0, 'balance': 100.0},
                                   {'type': 'withdraw', 'from': 1, 'to': None, 'amount': 50.0, 'balance': 50.0},
                                   {'type': 'withdraw', 'from': 1, 'to': 2, 'amount': 25.0, 'balance': 25.0}]],

]

def solutionBank(queries):
    bank = Bank()
    
    for idx_query, query in enumerate(queries):
        if query[0] == "CreateAccount":
            ans = bank.CreateAccount(query[1], query[2], query[3])
            trg = query[4]
        elif query[0] == "Deposit":
            ans = bank.Deposit(query[1], query[2])
            trg = query[3]
        elif query[0] == "GetBalance":
            ans = bank.GetBalance(query[1])
            trg = query[2]
        elif query[0] == "Withdraw":
            ans = bank.Withdraw(query[1], query[2])
            trg = query[3]
        elif query[0] == "Transfer":
            ans = bank.Transfer(query[1], query[2], query[3])
            trg = query[4]
        elif query[0] == "GetCustomerAccounts":
            ans = bank.GetCustomerAccounts(query[1])
            trg = query[2]
        elif query[0] == "GetAccountTransactions":
            ans = bank.GetAccountTransactions(query[1])
            trg = query[2]
        elif query[0] == "GetTopKAccounts":
            ans = bank.GetTopKAccounts(query[1])
            trg = query[2]        
        elif query[0] == "ScheduleTransfer":
            ans = bank.ScheduleTransfer(query[1], query[2], query[3])
            trg = query[4]        
        elif query[0] == "CancelTransfer":
            ans = bank.CancelTransfer(query[1], query[2], query[3])
            trg = query[4]        
        elif query[0] == "ExecuteScheduledTransfers":
            ans = bank.ExecuteScheduledTransfers()
            trg = query[1]        
        elif query[0] == "MergeAccounts":
            ans = bank.MergeAccounts(query[1], query[2])
            trg = query[3]        
        else:
            print(f"Invalid {idx_query=} {query=}")
            assert False
            return False

        if ans != trg:
            print(f"{idx_query=} {ans=} {trg=} {query=}")
            assert ans == trg
            return False
            
    return True

ret = solutionBank(queries)
print("All bank tests pass", ret)
