
def padovan(n):
    if n <= 2:
        return 1
    else:
        return padovan(n-2) + padovan(n-3)
    
if __name__ == "__main__":
    print(f"Padovan of 10 is {padovan(10)}")