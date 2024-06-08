#define MAX 75
#define PI 3.14159265358979323
#define M 15

double newcomb(double N, double L) {
    if (N == 0 || L == 0 || N == L) {
        return 0.0;
    }
    else{

        double comb = N * log(N) - L * log(L) - (N - L) * log(N - L) +
                      0.5 * log(2.0 * PI * N) - 0.5 * log(2.0 * PI * L) - 0.5 * log(2.0 * PI * (N - L));
        return comb;// - N*log(N)+N -.5*log(2*PI*N) + floor(.5*N)*log(floor(.5*N))-floor(.5*N) + .5*log(2*PI*floor(N*.5)) + ceil(.5*N)*log(ceil(.5*N))-ceil(.5*N) + .5*log(2*PI*ceil(N*.5));
        }
    }

double calcCombs(double NAi, double NAj, double la, double lA, double halpha, double ha, double ka,double km){
    double e = (halpha *la) + (ha *lA)  + (ka*la*lA) + (km*la*NAi);
    return newcomb(NAj,lA) + e;
}

__kernel void compute_Pijk(__global double* Pijk, const double halpha, const double ha,const double ka, const double km)
{
     int i = get_global_id(0);   //ARRAYindexes:[nai,naj,nak] [i,j,k]
     int nai = (i/(MAX*MAX));
     int naj = (i/MAX) % MAX;
     int nak = i%MAX;

     double loop = 0;   //a=lA, b=lalpha
     for(int a=0; a<=naj;a++)
     {
        int b = nak - a;
        if(0 <= b && b <= M)
        {
            loop +=  exp(calcCombs(nai,naj,b,a,halpha,ha,ka,km));
        }
     }
     Pijk[i] = loop;
}


