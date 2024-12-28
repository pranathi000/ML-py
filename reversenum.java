import java.util.*;
public class reversenum
{
    public static void main(String[] args)
    {
        Scanner sc=new Scanner(System.in);
        System.out.println("Enter the number");
        int num=sc.nextInt();
        int r,d=0;
        int temp=num;
        while(num>0)
        {
            r=num%10;
            d=d*10+r;
            num=num/10;
        }
        System.out.println(d+" is the reverse number of "+temp);
    }
}