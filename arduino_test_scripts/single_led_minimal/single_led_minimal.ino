#include <MD_MAX72xx.h>

#define HARDWARE_TYPE MD_MAX72XX::FC16_HW
#define MAX_DEVICES	16

#define CLK_PIN   13  // or SCK
#define DATA_PIN  11  // or MOSI
#define CS_PIN    10  // or SS

// SPI hardware interface
MD_MAX72XX mx = MD_MAX72XX(HARDWARE_TYPE, CS_PIN, MAX_DEVICES);

void led()
{
  int  r = 0, c = 32;
  mx.setPoint(r, c, true);
}

void setup()
{
  mx.begin();
  mx.clear();
}

void loop()
{
#if 1
  led();
#endif
}



