#include <iostream>

int padovan(int n)
{
    if (n <= 2)
    {
        return 1;
    }
    else
    {
        return padovan(n - 2) + padovan(n - 3);
    }
}

int main()
{
    std::cout << "Padovan of 10: " << padovan(10) << std::endl;
    return 0;
}