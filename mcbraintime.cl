#define MAX 25
#define PI 3.14159265358979323

double newcomb(double N, double L) {
    if (N == 0 || L == 0 || N == L) {
        return 0.0;}
    else{

    double comb = N * log(N) - L * log(L) - (N - L) * log(N - L) +
                      0.5 * log(2.0 * PI * N) - 0.5 * log(2.0 * PI * L) - 0.5 * log(2.0 * PI * (N - L));
    return comb;// - N*log(N)+N -.5*log(2*PI*N) + floor(.5*N)*log(floor(.5*N))-floor(.5*N) + .5*log(2*PI*floor(N*.5)) + ceil(.5*N)*log(ceil(.5*N))-ceil(.5*N) + .5*log(2*PI*ceil(N*.5));
        }
    }

double calcCombs(double NA, double NB, double la, double lA, double lb, double lB,double halpha, double ha,double hbeta, double hb, double kaa,double kab){
    double e = (halpha * (la-(MAX-NA)/2) + hbeta*(lb-(MAX-NB)/2) + ha * (lA-NA/2) + hb*(lB-NB/2) + kaa*((la-lA)*NA + (lb-lB)*NB) + kab*((lA-la)*NB + (lB-lb)*NA));
    return newcomb(MAX-NA,la) + newcomb(MAX-NB,lb)+ newcomb(NA, lA) + newcomb(NB, lB) + e;
}

__kernel void compute_Pkl(__global double* Pkl, const double halpha, const double ha,const double hbeta, const double hb, const double kaa, const double kab, const double NA, const double NB)
{
     int i = get_global_id(0);   //ARRAYindexes:[naj,nbl] [i,l]
     int naj = ((i)/(MAX+1));
     int nbl = (i)%(MAX+1);

     double loop4 = 0;   //a=lA, b=lB, c=la,d=lb
     for(int a=0; a<=NA;a++)
     {
        for(int b=0; b<=NB;b++)
        {
            int c = naj + a -NA; //lalpha
            int d = nbl + b -NB; //lbeta

            if(0 <= c && c <= (MAX-NA) && 0 <= d && d <= (MAX-NB))
            {
                loop4 +=  exp(calcCombs(NA,NB,c,a,d,b,halpha,ha,hbeta,hb,kaa,kab));
            }
        }

    }
    Pkl[i] = loop4;
}


