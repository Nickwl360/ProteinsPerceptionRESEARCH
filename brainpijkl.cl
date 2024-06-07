#define MAX 25
#define PI 3.14159265358979323

double stirling(double x){
    if(x ==0){
        return 0.0;
    }
    else{
        return x * log(x) - x + .5*log(2*PI*x);
    }
}

double newcomb(double N, double L) {
    int a = N;
    int b = L;
    double n = N;
    double l = L;


    if (a == 0) {
        return 0.0;
        }
    if (b == 0 || a == b){
        return (stirling(ceil(n/2)) + stirling(floor(n/2)) - stirling(n));
        }
    else{
        return (stirling(ceil(n/2)) + stirling(floor(n/2)) - stirling(l)- stirling(n-l));
        }
    }

double calcCombs(double NA, double NB, double la, double lA, double lb, double lB,double halpha, double ha,double hbeta, double hb, double kaa,double kab){
    double e = (halpha * (la-(MAX-NA)/2) + hbeta*(lb-(MAX-NB)/2) + ha * (lA-NA/2) + hb*(lB-NB/2) + kaa*((la-lA)*NA + (lb-lB)*NB) + kab*((lA-la)*NB + (lB-lb)*NA));
    double sum = newcomb(MAX-NA,la) + newcomb(MAX-NB,lb)+ newcomb(NA, lA) + newcomb(NB, lB) + e;
    return sum;
}

__kernel void compute_Pijkl(__global double* Pijkl, const double halpha, const double ha,const double hbeta, const double hb, const double kaa, const double kab)
{
     int i = get_global_id(0);   //ARRAYindexes:[nai,nbk,naj,nbl] [i,k,j,l]
     int size = 26;

     int nai = ((i)/((size)*(size)*(size)));
     int nbk = ((i)/((size)*(size)))%(size);
     int naj = ((i)/(size))%(size);
     int nbl = (i)%(size);

     double loop4 = 0;   //a=lA, b=lB, c=la,d=lb
     for(int a=0; a<=nai;a++)  //lA
     {
        for(int b=0; b<=nbk;b++)  //lB
        {
            int c = naj + a - nai; //lalpha
            int d = nbl + b - nbk; //lbeta

            if((0 <= c) && (c <= (MAX-nai)) && (0 <= d) && (d <= (MAX-nbk)))
            {
                loop4 +=  exp(calcCombs(nai,nbk,c,a,d,b,halpha,ha,hbeta,hb,kaa,kab));
            }
        }
    }
    Pijkl[i] = loop4;//i;//nai + nbk + naj + nbl;
}


