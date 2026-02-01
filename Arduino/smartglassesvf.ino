// CÔTÉ GAUCHE
// Pompe Gonflage Gauche (moteur 1)
#define POMPE_GONFLAGE_G_IN1  A0
#define POMPE_GONFLAGE_G_IN2  7
#define POMPE_GONFLAGE_G_PWM  A2

// Pompe Aspiration Gauche (moteur 2)
#define POMPE_ASPIRATION_G_IN1  12
#define POMPE_ASPIRATION_G_IN2  4
#define POMPE_ASPIRATION_G_PWM  3

// Valve Gauche
#define VALVE_GAUCHE_IN1  9
#define VALVE_GAUCHE_IN2  10

// CÔTÉ DROIT
// Pompe Gonflage Droite (moteur 3)
#define POMPE_GONFLAGE_D_IN1  8
#define POMPE_GONFLAGE_D_IN2  11
#define POMPE_GONFLAGE_D_PWM  6

// Pompe Aspiration Droite (moteur 4)
#define POMPE_ASPIRATION_D_IN1  5
#define POMPE_ASPIRATION_D_IN2  A1
#define POMPE_ASPIRATION_D_PWM  2

// Valve Droite
#define VALVE_DROITE_IN1  A3
#define VALVE_DROITE_IN2  A4

#define VITESSE_PWM 255
#define DELAI_STABILISATION 500

// Paramètres de l'asservissement
#define ZONE_MORTE            15   // Zone morte en Pa (pas d'action)
#define PWM_MIN               110    // PWM minimum pour démarrer la pompe
#define PWM_MAX               255   // PWM maximum
#define GAIN_PROPORTIONNEL    0.5 // Gain pour le calcul PWM
#define INTERVAL_LECTURE_MS   200  // Intervalle entre lectures capteur

// Capteur de pression (côté gauche uniquement)
#include <DFRobot_LWLP.h>
DFRobot_LWLP lwlp;

/*************** MODES ***************/
#define MODE_EMOTION 1
#define MODE_YOLO    2
#define MODE_NEUTRAL 3

// BOUTTONS
#define BOUTON_EMOTION 13
#define BOUTON_YOLO A1

int currentMode = MODE_NEUTRAL;
unsigned long lastButtonTime = 0;

// Variables asservissement YOLO

// APRÈS (inversion logique YOLO)
float consigne_pression_gauche = 0.0;   
float pression_simulee_gauche = 0.0;
float pression_simulee_droite = 0.0;

float consigne_pression_droite = 0.0; 


unsigned long derniere_action = 0;
unsigned long derniere_instruction_gauche = 0;
unsigned long derniere_instruction_droite = 0;
unsigned long dernier_increment_droite = 0;
unsigned long dernier_increment_gauche = 0;
const unsigned long TIMEOUT_INSTRUCTION = 2000;  // 2 secondes sans instruction
const unsigned long INTERVAL_INCREMENT_DROITE = 3000; // 10 secondes

bool asservissement_gauche_actif = false;
bool asservissement_droite_actif = false;

// ================== SETUP ==================
void setup() {
  Serial.begin(9600);
  // Serial.println("test");
  sendMode(currentMode);
  stopAll();

  pinMode(BOUTON_EMOTION, INPUT);
  pinMode(BOUTON_YOLO, INPUT);

  // Pompes gauche
  pinMode(POMPE_GONFLAGE_G_IN1, OUTPUT);
  pinMode(POMPE_GONFLAGE_G_IN2, OUTPUT);
  pinMode(POMPE_GONFLAGE_G_PWM, OUTPUT);

  pinMode(POMPE_ASPIRATION_G_IN1, OUTPUT);
  pinMode(POMPE_ASPIRATION_G_IN2, OUTPUT);
  pinMode(POMPE_ASPIRATION_G_PWM, OUTPUT);

  // Valve gauche
  pinMode(VALVE_GAUCHE_IN1, OUTPUT);
  pinMode(VALVE_GAUCHE_IN2, OUTPUT);

  // Pompes droite
  pinMode(POMPE_GONFLAGE_D_IN1, OUTPUT);
  pinMode(POMPE_GONFLAGE_D_IN2, OUTPUT);
  pinMode(POMPE_GONFLAGE_D_PWM, OUTPUT);

  pinMode(POMPE_ASPIRATION_D_IN1, OUTPUT);
  pinMode(POMPE_ASPIRATION_D_IN2, OUTPUT);
  pinMode(POMPE_ASPIRATION_D_PWM, OUTPUT);

  // Valve droite
  pinMode(VALVE_DROITE_IN1, OUTPUT);
  pinMode(VALVE_DROITE_IN2, OUTPUT);

  stopAll();
  // Serial.println("test_12");
  
  // Serial.println("test78");
  // Serial.println("Capteur LWLP initialise avec succes");
  
  sendMode(currentMode);
  derniere_instruction_gauche = millis();
  derniere_instruction_droite = millis();
  dernier_increment_droite = millis();
  dernier_increment_gauche = millis();
}

// ================== LOOP ==================
void loop() {

 

  readButtons();

  if (currentMode == MODE_EMOTION) {
    emotionLoop();
  } else if (currentMode == MODE_YOLO) {
    yoloLoop();
  }


}

// ================== BOUTONS & MODE ==================
void readButtons() {
  static int lastE = 0;
  static int lastY = 0;

  int E = digitalRead(BOUTON_EMOTION);
  int Y = digitalRead(BOUTON_YOLO);

  // Emotion button pressed → always go to EMOTION
  if (lastE == 0 && E == 1) {
    currentMode = MODE_EMOTION;
    sendMode(currentMode);
    stopAll();
  }

  // YOLO button pressed
  if (lastY == 0 && Y == 1) {
    if (currentMode == MODE_YOLO) {
      // Toggle OFF → neutral
      currentMode = MODE_NEUTRAL;
      asservissement_gauche_actif = false;
      asservissement_droite_actif = false;
    } else {
      // Toggle ON
      currentMode = MODE_YOLO;
      derniere_instruction_gauche = millis();
      derniere_instruction_droite = millis();
      dernier_increment_droite = millis();
      dernier_increment_gauche = millis();
    }
    sendMode(currentMode);
    stopAll();
  }

  lastE = E;
  lastY = Y;
}

void sendMode(int mode) {
  Serial.print("MODE:");
  Serial.println(mode);
}

// ================== MODE EMOTION ==================
void emotionLoop() {
    if (!Serial.available()) return;

    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
  
    if (cmd.startsWith("GG")) {
      int intensity = cmd.substring(3).toInt();
      gonflerGauche(intensity);
      delay(2000);
      stopAll();
    }
    else if (cmd.startsWith("AG")) {
      int intensity = cmd.substring(3).toInt();
      aspirerGauche(intensity);
      delay(2000);
      stopAll();
    }
    else if (cmd.startsWith("GD")) {
        int intensity = cmd.substring(3).toInt();
        gonflerDroite(intensity);
        delay(2000);
        stopAll();
    }
    else if (cmd.startsWith("AD")) {
        int intensity = cmd.substring(3).toInt();
        aspirerDroite(intensity);
        delay(2000);
        stopAll();
    }
    else if (cmd == "STOP") {
      stopAll();
    }
    else if (cmd == "DONE") {
      currentMode = MODE_NEUTRAL;
      sendMode(MODE_NEUTRAL);
      stopAll();
    }
}

// ================== MODE YOLO ==================
void yoloLoop() {
  // Lecture des commandes série
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("GG")) {
      int intensity = cmd.substring(3).toInt();
      // Intensité 0-100 → consigne 0 à -500
      consigne_pression_gauche = map(intensity, 0, 100, 0, -500);
      asservissement_gauche_actif = true;
      derniere_instruction_gauche = millis();
    }
    else if (cmd.startsWith("GD")) {
      int intensity = cmd.substring(3).toInt();
      // Intensité 0-100 → consigne 0 à -500
      consigne_pression_droite = map(intensity, 0, 100, 0, -500);
      asservissement_droite_actif = true;
      derniere_instruction_droite = millis();
    }
    else if (cmd == "STOP") {
      consigne_pression_gauche = 0;
      consigne_pression_droite = 0;
      asservissement_gauche_actif = true;
      asservissement_droite_actif = true;
      derniere_instruction_droite = millis();
      derniere_instruction_gauche = millis();
    }
  }

  // Vérification du timeout (2 secondes sans instruction)
  if (millis() - derniere_instruction_droite > TIMEOUT_INSTRUCTION) {
    if (consigne_pression_droite != 0) {
     
      consigne_pression_droite = 0;
   
      asservissement_droite_actif = true;
    }
  }

  if (millis() - derniere_instruction_gauche > TIMEOUT_INSTRUCTION) {
    if (consigne_pression_gauche != 0) {
      consigne_pression_gauche = 0;
     
      asservissement_gauche_actif = true;
    
    }
  }

  // Incrément de la pression simulée GAUCHE
  if (millis() - dernier_increment_gauche >= INTERVAL_INCREMENT_DROITE) {
    dernier_increment_gauche = millis();
    pression_simulee_gauche += 50;
    if (pression_simulee_gauche > 0) {
      pression_simulee_gauche = 0;
    }
  }

  if (millis() - dernier_increment_droite >= INTERVAL_INCREMENT_DROITE) {
    dernier_increment_droite = millis();
    pression_simulee_droite += 50;
    if (pression_simulee_droite > 0) {
      pression_simulee_droite = 0;
    }
  }


  // Asservissement périodique
  if (millis() - derniere_action >= INTERVAL_LECTURE_MS) {
    derniere_action = millis();
    


    if (asservissement_droite_actif) {
      asservir_pression_droite_simule();
    }

    // CÔTÉ GAUCHE - Simulation capteur
    if (asservissement_gauche_actif) {
      asservir_pression_gauche_simule();
    }



  }
}


void asservir_pression_gauche_simule() {
  float erreur = consigne_pression_gauche - pression_simulee_gauche;

  if (abs(erreur) <= ZONE_MORTE) {
    arreter_gauche();
  }
  else if (erreur < -ZONE_MORTE) {
    int pwm = calculer_pwm_proportionnel(abs(erreur));
    gonflerGaucheAsserv(pwm);
    pression_simulee_gauche -= 0.065 * INTERVAL_LECTURE_MS;
    if (pression_simulee_gauche < -500) pression_simulee_gauche = -500;
  }
  else if (erreur > ZONE_MORTE) {
    int pwm = calculer_pwm_proportionnel(abs(erreur));
    aspirerGaucheAsserv(pwm);
    pression_simulee_gauche += 0.15 * INTERVAL_LECTURE_MS;
    if (pression_simulee_gauche > 0) pression_simulee_gauche = 0;
  }
}

void asservir_pression_droite_simule() {
  float erreur = consigne_pression_droite - pression_simulee_droite;

  if (abs(erreur) <= ZONE_MORTE) {
    arreter_droite();
  }
  else if (erreur < -ZONE_MORTE) {
    int pwm = calculer_pwm_proportionnel(abs(erreur));
    gonflerDroiteAsserv(pwm);
    pression_simulee_droite -= 0.15 * INTERVAL_LECTURE_MS;
    if (pression_simulee_droite < -500) pression_simulee_droite = -500;
  }
  else if (erreur > ZONE_MORTE) {
    int pwm = calculer_pwm_proportionnel(abs(erreur));
    aspirerDroiteAsserv(pwm);
    pression_simulee_droite += 0.15 * INTERVAL_LECTURE_MS;
    if (pression_simulee_droite > 0) pression_simulee_droite = 0;
  }
}



// ================== CALCUL PWM ==================
int calculer_pwm_proportionnel(float erreur) {
  int pwm = (int)(abs(erreur) * GAIN_PROPORTIONNEL);
  
  if (pwm < PWM_MIN) {
    pwm = PWM_MIN;
  }
  if (pwm > PWM_MAX) {
    pwm = PWM_MAX;
  }
  
  return pwm;
}

// ================== FONCTIONS MATERIEL ASSERVISSEMENT ==================
void gonflerGaucheAsserv(int pwm) {
  digitalWrite(POMPE_ASPIRATION_G_IN1, HIGH);
  digitalWrite(POMPE_ASPIRATION_G_IN2, LOW);
  analogWrite(POMPE_ASPIRATION_G_PWM, pwm);

  digitalWrite(VALVE_GAUCHE_IN1, HIGH);
  digitalWrite(VALVE_GAUCHE_IN2, LOW);

  digitalWrite(POMPE_GONFLAGE_G_IN1, HIGH);
  digitalWrite(POMPE_GONFLAGE_G_IN2, LOW);
  analogWrite(POMPE_GONFLAGE_G_PWM, pwm);
}

void aspirerGaucheAsserv(int pwm) {
  digitalWrite(POMPE_GONFLAGE_G_IN1, LOW);
  digitalWrite(POMPE_GONFLAGE_G_IN2, LOW);
  analogWrite(POMPE_ASPIRATION_G_PWM, pwm);

  digitalWrite(VALVE_GAUCHE_IN1, LOW);
  digitalWrite(VALVE_GAUCHE_IN2, LOW);

  digitalWrite(POMPE_ASPIRATION_G_IN1, HIGH);
  digitalWrite(POMPE_ASPIRATION_G_IN2, LOW);
  analogWrite(POMPE_GONFLAGE_G_PWM, pwm);
}

void gonflerDroiteAsserv(int pwm) {
  digitalWrite(POMPE_ASPIRATION_D_IN1, HIGH);
  digitalWrite(POMPE_ASPIRATION_D_IN2, LOW);
  analogWrite(POMPE_ASPIRATION_D_PWM, pwm);

  digitalWrite(VALVE_DROITE_IN1, HIGH);
  digitalWrite(VALVE_DROITE_IN2, LOW);

  digitalWrite(POMPE_GONFLAGE_D_IN1, HIGH);
  digitalWrite(POMPE_GONFLAGE_D_IN2, LOW);
  analogWrite(POMPE_GONFLAGE_D_PWM, pwm);
}

void aspirerDroiteAsserv(int pwm) {
  digitalWrite(POMPE_GONFLAGE_D_IN1, LOW);
  digitalWrite(POMPE_GONFLAGE_D_IN2, LOW);
  analogWrite(POMPE_ASPIRATION_D_PWM, pwm);

  digitalWrite(VALVE_DROITE_IN1, LOW);
  digitalWrite(VALVE_DROITE_IN2, LOW);

  digitalWrite(POMPE_ASPIRATION_D_IN1, HIGH);
  digitalWrite(POMPE_ASPIRATION_D_IN2, LOW);
  analogWrite(POMPE_GONFLAGE_D_PWM, pwm);
}

void arreter_gauche() {
  analogWrite(POMPE_GONFLAGE_G_PWM, 0);
  analogWrite(POMPE_ASPIRATION_G_PWM, 0);
  digitalWrite(POMPE_GONFLAGE_G_IN1, LOW);
  digitalWrite(POMPE_GONFLAGE_G_IN2, LOW);
  digitalWrite(POMPE_ASPIRATION_G_IN1, LOW);
  digitalWrite(POMPE_ASPIRATION_G_IN2, LOW);
  digitalWrite(VALVE_GAUCHE_IN1, LOW);
  digitalWrite(VALVE_GAUCHE_IN2, LOW);
}

void arreter_droite() {
  analogWrite(POMPE_GONFLAGE_D_PWM, 0);
  analogWrite(POMPE_ASPIRATION_D_PWM, 0);
  digitalWrite(POMPE_GONFLAGE_D_IN1, LOW);
  digitalWrite(POMPE_GONFLAGE_D_IN2, LOW);
  digitalWrite(POMPE_ASPIRATION_D_IN1, LOW);
  digitalWrite(POMPE_ASPIRATION_D_IN2, LOW);
  digitalWrite(VALVE_DROITE_IN1, LOW);
  digitalWrite(VALVE_DROITE_IN2, LOW);
}

// ================== FONCTIONS MATERIEL MODE EMOTION ==================
void gonflerGauche(int intensity) {
  digitalWrite(POMPE_ASPIRATION_G_IN1, HIGH);
  digitalWrite(POMPE_ASPIRATION_G_IN2, LOW);
  analogWrite(POMPE_ASPIRATION_G_PWM, map(intensity, 0, 100, 0, VITESSE_PWM));

  digitalWrite(VALVE_GAUCHE_IN1, HIGH);
  digitalWrite(VALVE_GAUCHE_IN2, LOW);

  digitalWrite(POMPE_GONFLAGE_G_IN1, HIGH);
  digitalWrite(POMPE_GONFLAGE_G_IN2, LOW);
  analogWrite(POMPE_GONFLAGE_G_PWM, map(intensity, 0, 100, 0, VITESSE_PWM));
}

void aspirerGauche(int intensity) {
  digitalWrite(POMPE_GONFLAGE_G_IN1, LOW);
  digitalWrite(POMPE_GONFLAGE_G_IN2, LOW);
  analogWrite(POMPE_ASPIRATION_G_PWM, map(intensity, 0, 100, 0, VITESSE_PWM));

  digitalWrite(VALVE_GAUCHE_IN1, LOW);
  digitalWrite(VALVE_GAUCHE_IN2, LOW);

  digitalWrite(POMPE_ASPIRATION_G_IN1, HIGH);
  digitalWrite(POMPE_ASPIRATION_G_IN2, LOW);
  analogWrite(POMPE_GONFLAGE_G_PWM, map(intensity, 0, 100, 0, VITESSE_PWM));
}

void gonflerDroite(int intensity) {
  digitalWrite(POMPE_ASPIRATION_D_IN1, HIGH);
  digitalWrite(POMPE_ASPIRATION_D_IN2, LOW);
  analogWrite(POMPE_ASPIRATION_D_PWM, map(intensity, 0, 100, 0, VITESSE_PWM));

  digitalWrite(VALVE_DROITE_IN1, HIGH);
  digitalWrite(VALVE_DROITE_IN2, LOW);

  digitalWrite(POMPE_GONFLAGE_D_IN1, HIGH);
  digitalWrite(POMPE_GONFLAGE_D_IN2, LOW);
  analogWrite(POMPE_GONFLAGE_D_PWM, map(intensity, 0, 100, 0, VITESSE_PWM));
}

void aspirerDroite(int intensity) {
  digitalWrite(POMPE_GONFLAGE_D_IN1, LOW);
  digitalWrite(POMPE_GONFLAGE_D_IN2, LOW);
  analogWrite(POMPE_ASPIRATION_D_PWM, map(intensity, 0, 100, 0, VITESSE_PWM));

  digitalWrite(VALVE_DROITE_IN1, LOW);
  digitalWrite(VALVE_DROITE_IN2, LOW);

  digitalWrite(POMPE_ASPIRATION_D_IN1, HIGH);
  digitalWrite(POMPE_ASPIRATION_D_IN2, LOW);
  analogWrite(POMPE_GONFLAGE_D_PWM, map(intensity, 0, 100, 0, VITESSE_PWM));
}

void stopAll() {
  analogWrite(POMPE_GONFLAGE_G_PWM, 0);
  analogWrite(POMPE_ASPIRATION_G_PWM, 0);
  analogWrite(POMPE_GONFLAGE_D_PWM, 0);
  analogWrite(POMPE_ASPIRATION_D_PWM, 0);

  digitalWrite(POMPE_GONFLAGE_G_IN1, LOW);
  digitalWrite(POMPE_GONFLAGE_G_IN2, LOW);
  digitalWrite(POMPE_ASPIRATION_G_IN1, LOW);
  digitalWrite(POMPE_ASPIRATION_G_IN2, LOW);

  digitalWrite(POMPE_GONFLAGE_D_IN1, LOW);
  digitalWrite(POMPE_GONFLAGE_D_IN2, LOW);
  digitalWrite(POMPE_ASPIRATION_D_IN1, LOW);
  digitalWrite(POMPE_ASPIRATION_D_IN2, LOW);

  digitalWrite(VALVE_GAUCHE_IN1, LOW);
  digitalWrite(VALVE_GAUCHE_IN2, LOW);
  digitalWrite(VALVE_DROITE_IN1, LOW);
  digitalWrite(VALVE_DROITE_IN2, LOW);
}
