#include <MD_MAX72xx.h>

#define HARDWARE_TYPE MD_MAX72XX::FC16_HW
#define MAX_DEVICES	16

#define CLK_PIN   13  // or SCK
#define DATA_PIN  11  // or MOSI
#define CS_PIN    10  // or SS

// SPI hardware interface
MD_MAX72XX mx = MD_MAX72XX(HARDWARE_TYPE, CS_PIN, MAX_DEVICES);

void led_square_raster(int c_start, int r_start, int edge_length, int led_delay)
{
  // central 4 squares (r: 0-7, c: 40 - 55, 72-87)
  int c = c_start;
  int r = r_start;
  int maxR = 8; 
  int maxC = c_start + edge_length; 
  int dC = 1; 
  int dR = 1;
  
  for (int i=0; i<edge_length*edge_length; i++){
    mx.setPoint(r, c, false);
    c += dC; 
    mx.setPoint(r, c, true);
    // triggerCamera();
    delay(led_delay); 

    
    if (c == maxC)
    {
      mx.clear();
      c = c_start;
      r += dR; 
      
    } 
    if (r == maxR)
    {
      mx.clear();
      r = 0;
      c-= 32;
      c_start -= 32;
      maxC -= 32;
    } 


  }

}

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
  mx.begin();
  mx.clear();
  // led_square_raster(71, 0, 15, 50);
}

void loop()
{
#if 1
    led_single(111, 7);
//  led_square_raster();
#endif
}
