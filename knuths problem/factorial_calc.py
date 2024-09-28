
def factorial(n):
    temp = []         #list that will contain large int values
    current = ans = 1
    while current <= n:
        while ans < pow(10, 50): #adjust value if n > 1,000,000
            ans = ans*current
            current += 1
            if current > n:
                break
    temp.append(str(ans))
    ans = 1
    answer = list(temp[-1])[::-1]#to store intermediate values of answer
    temp.pop()
    i = len(temp)
    while i > 0:
        answer = mult(answer, int(temp[-1]))
        temp.pop()
        i -= 1

    result = ''.join(answer[::-1])
    return(int(result))
 
 
def mult(answer, num):
    i, maxlen, carry = 0, len(answer), 0
    while i < maxlen:
        temp = num*int(answer[i])+carry
        answer[i] = str(temp%10)
        carry = temp//10
        i += 1
    if carry != 0:
        answer = answer+list(str(carry)[::-1])
    return answer

