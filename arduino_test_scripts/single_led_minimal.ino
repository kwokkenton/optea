#include <MD_MAX72xx.h>

#define HARDWARE_TYPE MD_MAX72XX::FC16_HW
#define MAX_DEVICES	16

#define CLK_PIN   13  // or SCK
#define DATA_PIN  11  // or MOSI
#define CS_PIN    10  // or SS

// SPI hardware interface
MD_MAX72XX mx = MD_MAX72XX(HARDWARE_TYPE, CS_PIN, MAX_DEVICES);
// We always wait a bit between updates of the display
#define  DELAYTIME  100  // in milliseconds

void main()
{
  const int minC = 0;
  const int maxC = mx.getColumnCount()-1;
  const int minR = 0;
  const int maxR = ROW_SIZE-1;

  int  nCounter = 0;

  int  r = 0, c = 32;

  mx.clear();

  while (nCounter++ < 1000)
  {
    mx.setPoint(r, c, true);
    delay(DELAYTIME/2);
  }
}

void setup()
{
  mx.begin();
}

void loop()
{
  main();
}



