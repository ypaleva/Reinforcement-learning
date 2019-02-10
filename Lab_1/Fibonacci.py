import time


def F(n):
    if n <= 1:
        return n
    else:
        return F(n - 1) + F(n - 2)


past_fib = {}


def DP_F(n):
    if n in past_fib:
        return past_fib[n]
    if n <= 1:
        past_fib[n] = n
        return n
    total = DP_F(n - 1) + DP_F(n - 2)
    past_fib[n] = total
    return total

arr_fib = []
answers_fib = []
for x in range(1, 31):
    start = time.time()
    ans = F(x)
    print(ans)
    end = time.time()
    delta = end - start
    arr_fib.append(delta)
    answers_fib.append(ans)
print('Naive recursive version: ')
print(arr_fib)
print(answers_fib)

arr_dp_fib = []
answers_dp_fib = []
for x in range(1, 31):
    start = time.time()
    ans = DP_F(x)
    end = time.time()
    delta = end - start
    arr_dp_fib.append(delta)
    answers_dp_fib.append(ans)
print('Dynamic programming: ')
print(arr_dp_fib)
print(answers_dp_fib)



