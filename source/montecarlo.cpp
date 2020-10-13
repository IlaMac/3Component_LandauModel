#include "montecarlo.h"
#include "main.h"
#include "rng.h"
#include "class_tic_toc.h"

void metropolis( struct Node* Site, struct MC_parameters &MCp, struct H_parameters &Hp,  double my_beta){

    double d_theta, d_A, rand;
    unsigned int ix, iy, iz, alpha, vec, i;
    double acc_rate=0.5, acc_A=0., acc_theta=0.; // acc_rho=0.,
    struct O2 NewPsi{};
    struct O2 OldPsi{};
    double NewA, OldA;
    double newE, oldE, minus_deltaE;
    double h3=(Hp.h*Hp.h*Hp.h);
    class_tic_toc t_localHtheta(true,5,"local_Htheta");
    class_tic_toc t_localHA(true,5,"local_HA");


    for (iz= 0; iz < Lz; iz++) {
        for (iy = 0; iy < Ly; iy++) {
            for (ix = 0; ix < Lx; ix++) {
                i = ix + Lx * (iy + iz * Ly);

                /*******PHASE ONLY UPDATE**************/
                for (alpha = 0; alpha < 3; alpha++) {
                    OldPsi = Site[i].Psi[alpha];
                    //t_localHtheta.tic();
                    oldE = local_Htheta(OldPsi, ix, iy, iz, alpha, Hp, Site);
                    //t_localHtheta.toc();
                    d_theta = rn::uniform_real_box(-MCp.lbox_theta, MCp.lbox_theta);
                    NewPsi.t = fmod(OldPsi.t + d_theta, C_TWO_PI);
                    NewPsi.r=1;
                    //NewPsi.r = OldPsi.r;
                    //polar_to_cartesian(NewPsi);
                    //t_localHtheta.tic();
                    newE = local_Htheta(NewPsi, ix, iy, iz, alpha, Hp, Site);
                    //t_localHtheta.toc();
                    minus_deltaE = h3 * (oldE - newE);
                    if (minus_deltaE > 0) {
                        Site[i].Psi[alpha] = NewPsi;
                        acc_theta++;
                    } else {
                        rand = rn::uniform_real_box(0, 1);
                        //Boltzmann weight: exp(-\beta E) E= h³ \sum_i E(i)
                        if (rand < exp(my_beta * minus_deltaE)) {
                            Site[i].Psi[alpha] = NewPsi;
                            acc_theta++;
                        }
                    }
                }
                if (Hp.e != 0) {
                    /**********VECTOR POTENTIAL UPDATE********/
                    for (vec = 0; vec < 3; vec++) {
                        //Update of A
                        OldA = Site[i].A[vec];
                        //t_localHA.tic();
                        oldE = local_HA(OldA, ix, iy, iz, vec, Hp, Site);
                        //t_localHA.toc();
                        d_A = rn::uniform_real_box(-MCp.lbox_A, MCp.lbox_A);
                        NewA = OldA + d_A;
                        //t_localHA.tic();
                        newE = local_HA(NewA, ix, iy, iz, vec, Hp, Site);
                        //t_localHA.toc();
                        minus_deltaE = h3 * (oldE - newE);
                        if (minus_deltaE > 0.) {
                            Site[i].A[vec] = NewA;
                            acc_A++;
                        } else {
                            rand = rn::uniform_real_box(0, 1);
                            //Boltzmann weight: exp(-\beta E) E= h³ \sum_i E(i)
                            if (rand < exp(my_beta * minus_deltaE)) {
                                Site[i].A[vec] = NewA;
                                acc_A++;

                            }
                        }
                    }
                }
            }
        }
    }
//    t_localHA.print_measured_time();
 //   t_localHtheta.print_measured_time();
 //   t_localHpsi.print_measured_time();
    acc_theta=(double) acc_theta/(3*N);
    acc_A=(double) acc_A/(3*N);
    MCp.lbox_theta= MCp.lbox_theta*(acc_theta/acc_rate);
    MCp.lbox_A= MCp.lbox_A*(acc_A/acc_rate);
}


double local_Htheta(struct O2 Psi, unsigned int ix, unsigned int iy, unsigned int iz, unsigned int alpha,  struct H_parameters &Hp, struct Node* Site){

    double h_Kinetic=0., h_Josephson=0., h_AB=0., h_tot;
    double h2=(Hp.h*Hp.h);
    double J_alpha1, J_beta1, J_alpha2, J_beta2;
    double gauge_phase1, gauge_phase2;
    unsigned int beta, vec, i;

    i=ix +Lx*(iy+Ly*iz);
    //We need to compute just the part of the Hamiltonian involving Psi.t

    //Kinetic= -(1/h²)*\sum_k=1,2,3 (|Psi_{alpha}(r)||Psi_{alpha}(r+k)|* cos(theta_{alpha}(r+k) - theta_{alpha}(r) +h*e*A_k(r))) + (|Psi_{alpha}(r-k)||Psi_{alpha}(r)|* cos(theta_{alpha}(r) - theta_{alpha}(r-k) +h*e*A_k(r-k)))
    for(vec=0; vec<3; vec++){

        gauge_phase1=Site[nn(i, vec, 1)].Psi[alpha].t - Psi.t + Hp.h*Hp.e*Site[i].A[vec];
        gauge_phase2=Psi.t -Site[nn(i, vec, -1)].Psi[alpha].t + Hp.h*Hp.e*Site[nn(i, vec, -1)].A[vec];
        h_Kinetic-=(1./h2)*cos(gauge_phase1);
        h_Kinetic-=(1./h2)*cos(gauge_phase2);

       //Andreev-Bashkin term = \sum_beta!=alpha \sum_k=1,2,3 -nu*J^k_alpha*J^k_beta;
        // with J^k_alpha= |Psi_{alpha}(r)||Psi_{alpha}(r+k)|* sin(theta_{alpha}(r+k) - theta_{alpha}(r) +h*e*A_k(r)))
        if(Hp.nu !=0 ) {

        J_alpha1= (1./Hp.h)*sin(gauge_phase1);
        J_alpha2= (1./Hp.h)*sin(gauge_phase2);

        for (beta = 0; beta < 3; beta++) {
                if (beta != alpha) {
                J_beta1=(1./Hp.h)*sin(Site[nn(i, vec, 1)].Psi[beta].t - Site[i].Psi[beta].t + Hp.h*Hp.e*Site[i].A[vec]);
                h_AB -= Hp.nu *(J_alpha1*J_beta1);
                J_beta2=(1./Hp.h)*sin( Site[i].Psi[beta].t -Site[nn(i, vec, -1)].Psi[beta].t + Hp.h*Hp.e*Site[nn(i, vec, -1)].A[vec]);
                h_AB -= Hp.nu * (J_alpha2 *J_beta2);
                	}
        	}
        }
    }

    //Josephson= eta* \sum_beta!=alpha |Psi_{alpha}(r)||Psi_{beta}(r)|* cos(theta_{alpha}(r) - theta_{beta}(r))
    for(beta=0; beta<3; beta++){
	if(beta != alpha) {
            h_Josephson += (Hp.eta * cos(Psi.t- Site[i].Psi[beta].t));
        }
    }

    h_tot= h_Kinetic + h_Josephson + h_AB;
    return h_tot;
}

double local_HA(double A, unsigned int ix, unsigned int iy, unsigned int iz,  unsigned int vec,  struct H_parameters &Hp, struct Node* Site){

    double h_Kinetic=0., h_B, h_AB=0., h_tot;
    double h2=(Hp.h*Hp.h);
    double J_alpha, J_beta, J_gamma;
    unsigned int alpha, i;
    std::vector<struct O2> aux_field(3);
    i=ix +Lx*(iy+Ly*iz);

    for(alpha=0; alpha<3; alpha++) {
        aux_field[alpha].r =Site[nn(i, vec, 1)].Psi[alpha].r;
        aux_field[alpha].t =Site[nn(i, vec, 1)].Psi[alpha].t + Hp.h * Hp.e * A;
        polar_to_cartesian(aux_field[alpha]);
    }

    //Compute the local Energy respect to a given component (alpha) of the vector potential A and a given spatial position (r=(ix, iy, iz))
    //We need to compute just the part of the Hamiltonian involving A

    //Kinetic= -(1/h²)*\sum_k=1,2,3 |Psi_{alpha}(r)||Psi_{alpha}(r+k)|* cos(theta_{alpha}(r+k) - theta_{alpha}(r) +h*e*A_k(r))
    for(alpha=0; alpha<3; alpha++) {
        h_Kinetic -= (1. / h2) * O2prod(Site[i].Psi[alpha], aux_field[alpha]);
	 }
    //Andreev-Bashkin term = \sum_beta!=alpha \sum_k=1,2,3 -nu*(J^k_alpha*J^k_beta);
    // with J^k_alpha= |Psi_{alpha}(r)||Psi_{alpha}(r+k)|* sin(theta_{alpha}(r+k) - theta_{alpha}(r) +h*e*A_k(r)))
    if(Hp.nu !=0 ) {

        J_alpha=(1. / Hp.h) * O2vprod(Site[i].Psi[0], aux_field[0]);
        J_beta=(1. / Hp.h) * O2vprod(Site[i].Psi[1], aux_field[1]);
        J_gamma=(1. / Hp.h) * O2vprod(Site[i].Psi[2], aux_field[2]);

            h_AB -= Hp.nu * ((J_alpha *J_beta)  + (J_alpha*J_gamma) + (J_gamma* J_beta));
    }

    h_B=(0.5/h2)*F_2(A, vec, ix, iy, iz, Site);

    h_tot= h_Kinetic + h_AB+ h_B;
    return h_tot;
}

double F_2(double newA, unsigned int k, unsigned int ix, unsigned int iy, unsigned int iz, struct Node* Site){

    unsigned int l;
    unsigned int i;
    double F2_A=0., F_A;
    i=ix +Lx*(iy+Ly*iz);
    //All the plaquettes involving A_vec(i)
    for(l=0; l<3; l++){
        if(l!= k){
            F_A=(newA + Site[nn(i, k, 1)].A[l] - Site[nn(i, l, 1)].A[k] - Site[i].A[l]);
            F2_A+=(F_A*F_A);
        }
    }
    for(l=0; l<3; l++){
        if(l!= k){
            F_A=(Site[nn(i, l, -1)].A[k]+ Site[nn(nn(i, l, -1), k, 1)].A[l] - newA - Site[nn(i, l, -1)].A[l]);
            F2_A+=(F_A*F_A);
        }
    }

    return F2_A;
}

