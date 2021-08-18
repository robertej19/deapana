{
ifstream file;
//file.open("./lund5/aao_norad1.lund");
file.open("pi0_gen1824.lund");

float a[10];
string str;
TLorentzVector beam;
TLorentzVector escat;
TLorentzVector q;

beam.SetXYZM(0,0,10.6014,0);
Double_t Q2;
Double_t Nu;
Double_t Xb;
Double_t Epx;
Double_t Epy;
Double_t Epz;
Double_t Etheta;


Double_t Ppx;
Double_t Ppy;
Double_t Ppz;
Double_t Pp;
Double_t Ptheta;

Double_t Gpx[2];
Double_t Gpy[2];
Double_t Gpz[2];
Double_t Gp[2];
Double_t Gtheta[2];


    TLorentzVector p4_proton;
    TLorentzVector p4_electron;
    TLorentzVector p4_gamma[2];
    TLorentzVector p4_pi0;
    TLorentzVector p4_beam;
    TLorentzVector p4_target;

    p4_beam.SetXYZM(0,0,10.604,0);
    p4_target.SetXYZM(0,0,0,0.938);

    Float_t Pi0M;


//ofstream file1;
//file1.open("out.txt", std::ofstream::app);

TFile *rFile = TFile::Open("xB.root","RECREATE");
TTree *T=new TTree("T","");

T->Branch("Pi0M",&Pi0M,"Pi0/F");
T->Branch("xB",&Xb,"xB/D");
T->Branch("Q2",&Q2,"Q2/D");
T->Branch("Nu",&Nu,"Nu/D");
T->Branch("Epx",&Epx,"Epx/D");
T->Branch("Epy",&Epy,"Epy/D");
T->Branch("Epz",&Epz,"Epz/D");
T->Branch("Etheta",&Etheta,"Etheta/D");

T->Branch("Ppx",&Ppx,"Ppx/D");
T->Branch("Ppy",&Ppy,"Ppy/D");
T->Branch("Ppz",&Ppz,"Ppz/D");
T->Branch("Pp",&Pp,"Pp/D");
T->Branch("Ptheta",&Ptheta,"Ptheta/D");

T->Branch("Gpx",&Gpx,"Gpx[2]/D");
T->Branch("Gpy",&Gpy,"Gpy[2]/D");
T->Branch("Gpz",&Gpz,"Gpz[2]/D");
T->Branch("Gp",&Gp,"Gp[2]/D");
T->Branch("Gtheta",&Gtheta,"Gtheta[2]/D");


Float_t misPi2;
T->Branch("misPi2",&misPi2,"misPi2/F");
int indx =0;
int g1=0;

bool fE = false;
bool fP = false;
bool fG = false;
while(getline(file,str)){
	cout<<" index = "<<indx<<endl;
	sscanf(str.c_str(),"%f\%f\%f\%f\%f\%f\%f\%f\%f\%f",&a[0],&a[1],&a[2],&a[3],&a[4],&a[5],&a[6],&a[7],&a[8],&a[9]);

	if(indx%5 == 0) {indx++; continue;}

	indx++;
	if( ( (int)(a[3]) ) == 11){
		escat.SetXYZM(a[6],a[7],a[8],0);
		q = beam - escat;
		Q2 = -q.M2();
		Nu = q.E();
		Xb = Q2/(2*0.938*Nu);
		Epx = a[6];
		Epy = a[7];
		Epz = a[8];
		Etheta = TMath::ATan2(TMath::Sqrt(Epx*Epx+Epy*Epy) ,Epz);
	//	file1<<Xb<<endl;
		fE = true;

		p4_electron.SetXYZM(Epx,Epy,Epz,0.000511);
	}
	if( ( (int)(a[3]) ) == 22){
		if(g1==0){
        	        Gpx[0] = a[6];
	                Gpy[0] = a[7];
                	Gpz[0] = a[8];
        	        Gp[0] = TMath::Sqrt( Gpx[0]*Gpx[0] + Gpy[0]*Gpy[0] + Gpz[0]*Gpz[0]);
	                Gtheta[0] = TMath::ATan2(TMath::Sqrt(Gpx[0]*Gpx[0]+Gpy[0]*Gpy[0]) ,Gpz[0]);
			p4_gamma[0].SetXYZM(Gpx[0],Gpy[0],Gpz[0],0);
			g1++;
		}
		else{
                        Gpx[1] = a[6];
                        Gpy[1] = a[7];
                        Gpz[1] = a[8];
                        Gp[1] = TMath::Sqrt( Gpx[1]*Gpx[1] + Gpy[1]*Gpy[1] + Gpz[1]*Gpz[1]);
                        Gtheta[1] = TMath::ATan2(TMath::Sqrt(Gpx[1]*Gpx[1]+Gpy[1]*Gpy[1]) ,Gpz[1]);
			fG = true;
                        p4_gamma[1].SetXYZM(Gpx[1],Gpy[1],Gpz[1],0);

			g1 = 0;
		}

	}
	if(( (int)(a[3]) ) == 2212){
                Ppx = a[6];
                Ppy = a[7];
                Ppz = a[8];
		Pp = TMath::Sqrt( Ppx*Ppx + Ppy*Ppy + Ppz*Ppz);
                Ptheta = TMath::ATan2(TMath::Sqrt(Ppx*Ppx+Ppy*Ppy) ,Ppz);
		fP = true;
		p4_proton.SetXYZM(Ppx,Ppy,Ppz,0.938);
	}
	if( fE && fP && fG){
		p4_pi0 = p4_gamma[0] + p4_gamma[1];
		Pi0M = p4_pi0.M();


                p4_beam.SetXYZM(0,0,10.6,0);
                p4_target.SetXYZM(0,0,0,0.938);
                p4_electron.SetXYZM(Epx,Epy,Epz,0);
                p4_proton.SetXYZM(Ppx,Ppy,Ppz,0.938);

                misPi2 = (p4_beam + p4_target - p4_electron - p4_proton).M2();

		T->Fill();
		fE = false;
		fP = false;
		fG = false;
	}	

}

rFile->Write();
rFile->Close();
//file1.close();
}
