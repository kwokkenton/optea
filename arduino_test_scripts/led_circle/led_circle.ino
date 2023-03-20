#include <MD_MAX72xx.h>

#define HARDWARE_TYPE MD_MAX72XX::FC16_HW
#define MAX_DEVICES	16

#define CLK_PIN   13  // or SCK
#define DATA_PIN  11  // or MOSI
#define CS_PIN    10  // or SS

// SPI hardware interface
MD_MAX72XX mx = MD_MAX72XX(HARDWARE_TYPE, CS_PIN, MAX_DEVICES);



void triggerCamera(int cam_del)
{
  digitalWrite(7, HIGH);
  delay(cam_del);
  digitalWrite(7,LOW);
}



void led_circle(float c_0, float r_0, int radius, int led_delay, int cam_del)
{
  // loop through all leds, switch on if outside defined radius
  for (int j=0; j<32; j++){
    for (int i=0; i<32; i++){
      int rad_sq = sq(i-c_0) + sq(j-r_0);
      if (rad_sq >= sq(radius)){
         mx.setPoint((7- j % 8), (i + 32*(j/8)), true);
//         triggerCamera(cam_del);
         delay(led_delay);
//         mx.setPoint((7- j % 8), (i + 32*(j/8)), false);
    }
  }
}
}

void led_all_raster(int led_delay, int cam_del)
{
  // loop through all leds, switch on if outside defined radius
  for (int j=8; j<24; j++){ // FIX THIS!!!!
    for (int i=8; i<24; i++){
       mx.setPoint((7- j % 8), (i + 32*(j/8)), true);
       triggerCamera(cam_del);
       delay(led_delay);
       mx.setPoint((7- j % 8), (i + 32*(j/8)), false);
  }
}
}



void led_square_raster(int led_delay, int cam_del)
{
  // loop through all leds, switch on if outside defined radius
  for (int j=8; j<24; j++){
    for (int i=8; i<24; i++){
       mx.setPoint((7- j % 8), (i + 32*(j/8)), true);
       triggerCamera(cam_del);
       delay(led_delay);
       mx.setPoint((7- j % 8), (i + 32*(j/8)), false);
  }
}
}

void led_circle_inside(float c_0, float r_0, int radius, int led_delay, int cam_del)
{
  // loop through all leds, switch on if outside defined radius
  for (int j=0; j<32; j++){
    for (int i=0; i<32; i++){
      int rad_sq = sq(i-c_0) + sq(j-r_0);
      if (rad_sq < sq(radius)){
         mx.setPoint((7- j % 8), (i + 32*(j/8)), true);
//         triggerCamera(cam_del);
//         delay(led_delay);
//         mx.setPoint((7- j % 8), (i + 32*(j/8)), false);
    }
  }
}
}


void central_leds(int i, int j)
{
  mx.setPoint((7- j % 8), (i + 32*(j/8)), true);
}


void half_leds_on(int radius, int half)
{
  for (int j=0; j<32; j++){
    for (int i=0; i<32; i++){
      int rad_sq = sq(i-15.5) + sq(j-15.5);
      if (half == 3){ // left
        if (rad_sq < sq(radius) && j<=15){
        mx.setPoint((7- j % 8), (i + 32*(j/8)), true);
        }}
      if (half == 4){ // right
        if (rad_sq < sq(radius) && j>15){
        mx.setPoint((7- j % 8), (i + 32*(j/8)), true);
        }}
      if (half == 1){ // top
        if (rad_sq < sq(radius) && i<=15){
        mx.setPoint((7- j % 8), (i + 32*(j/8)), true);
        }}
      if (half == 2){ // bottom
        if (rad_sq < sq(radius) && i>15){
        mx.setPoint((7- j % 8), (i + 32*(j/8)), true);
        }} 
}
}}
  

//void led_square_raster(int c_0, int r_0, int edge_length, int led_delay)
//{
//  // central 4 squares (r: 0-7, c: 40 - 55, 72-87)
//  int c = c_0;
//  int r = r_0;
//  int maxR = 8; 
//  int maxC = c_0 + edge_length; 
//  int dC = 1; 
//  int dR = 1;
//  
//  for (int i=0; i<edge_length*edge_length; i++){
//    mx.setPoint(r, c, false);
//    c += dC; 
//    mx.setPoint(r, c, true);
//    // triggerCamera();
//    delay(led_delay); 
//
//    
//    if (c == maxC)
//    {
//      mx.clear();
//      c = c_0;
//      r += dR; 
//      
//    } 
//    if (r == maxR)
//    {
//      mx.clear();
//      r = 0;
//      c-= 32;
//      c_0 -= 32;
//      maxC -= 32;
//    } 
//
//
//  }
//
//}

void led_single(int c, int r)
{
  // central 4 squares (r: 0-7, c: 40 - 55, 72-87)
  // int  r = 7, c = 87;
  while (1){
      mx.setPoint(r, c, true);
    //delay(1000); 
  }

}

void setup()
{
  pinMode(7, OUTPUT);
  int led_delay = 5;
  mx.begin();
  delay(led_delay); 
  mx.clear();
  delay(led_delay); 
  // central_leds(15, 15);
//  central_leds(15, 16);
//  central_leds(16, 15);
//  central_leds(16, 16);
  // led_square_raster(71, 0, 15, 50);
//  led_square_raster(300, 200);
//half_leds_on(5, 2);
  led_circle_inside(15.5, 15.5, 5, 5, 2000);
//  mx.setPoint((7- 16 % 8), (16 + 32*(16/8)), true);
//  triggerCamera(2);

}

void loop()
{
    delay(100000); 
}
