#define M 3
#define MAXTOP 6
#define MAXBOT 26
#define PI 3.14159265

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

double calcP(double NA, double NB, double NC, double ND, double la,double lA,double lb, double lB, double lgamma, double lc, double ldelta, double ld,double halpha, double hA, double hbeta, double hB, double hgamma,double hdelta, double hc, double hd, double kcoop,double kcomp, double kdu, double kud, double kx){
    double e = (halpha * (la-((MAXTOP-NA)/2)) + hbeta * (lb - ((MAXTOP-NB)/2)) + hgamma*(lgamma-((MAXBOT-NC))/2) + hdelta*(ldelta-(MAXBOT-ND)/2) + hA * (lA-NA/2) + hB*(lB-NB/2) + hc*(lc-NC/2)+hd*(ld-ND/2) + kcoop*((la-lA)*NA - (MAXTOP*NA/2) + (lb-lB)*NB- (MAXTOP*NB/2)) + kcomp*((lA-la)*NB -(MAXTOP*NB/2) + (lB-lb)*NA - (MAXTOP*NA/2)) + kdu*((la-lA)*NC - (NC*MAXTOP/2) +(lb-lB)*ND -(ND*MAXTOP/2)) + kud*(NA*(lc-lgamma)-(NA*MAXBOT/2)+NB*(ld-ldelta)-(NB*MAXBOT/2))+kx*((lB-lb)*NC-(NC*MAXTOP/2)+(lA-la)*ND-(ND*MAXTOP/2)));
    double sum = (newcomb(MAXTOP-1-NA,la) + newcomb(MAXTOP-1-NB,lb)+ newcomb(NA, lA) + newcomb(NB, lB) + newcomb(MAXBOT-1-NC,lgamma) + newcomb(NC,lc) + newcomb(MAXBOT-1-ND,ldelta) + newcomb(ND,ld) + e);
    return sum;
}


__kernel void compute_Pmnop(__global double* Pmnop, const double NA, const double NB, const double NC, const double ND, const double halpha, const double hbeta, const double hgamma, const double hdelta,const double ha,const double hb, const double hc, const double hd, const double kcoop,const double kcomp,const double kdu, const double kup,const double kx)
{
     int i = get_global_id(0);   //ARRAYindexes:[na2,nb2,nc2,nd2] [maxtop,maxtop,maxbot,maxbot]

     int na2 = (i/(MAXTOP*MAXBOT*MAXBOT));
     int nb2 = (i/(MAXBOT*MAXBOT))%MAXTOP;
     int nc2 = (i/(MAXBOT))%MAXBOT;
     int nd2 = (i)%MAXBOT;

     double loop=0;   //a=lA, b=lB, c=lC,d=lD, e=la,f=lb,g=lc,h=ld
     for(int a=0; a<=MAXTOP-1; a++)
     {
        for(int b=0; b<=MAXTOP-1; b++)
        {
            for (int c=0; c<=M;c++)
            {
                for(int d=0; d<=M;d++)
                {
                    int e = na2 - NA + a;  //lalpha
                    int f = nb2 - NB + b;  //lbeta
                    int g = nc2 - NC + c;  //lgamma
                    int h = nd2 - ND + d; // ldelta
                    if(0 <= e && e <= (MAXTOP-1) && 0 <= f && f <= (MAXTOP-1) && 0 <= g && g <= (M) && 0 <= h && h <= (M))
                       //&& (0<=nam) && nam<=(MAXTOP-1) && (0<=nbn) && nbn<=(MAXTOP-1) && (0<=nco) && nco<=(MAXBOT-1) && (0<=ndp) && ndp<=(MAXBOT-1)
                    {
                        loop +=  exp(calcP(NA,NB,NC,ND, e,a,f,b,g,c,h,d, halpha,ha,hbeta,hb,hgamma,hdelta,hc,hd,kcoop,kcomp,kdu,kup,kx));
                    }
                }

            }


        }

    }
    Pmnop[i] = loop;
}



