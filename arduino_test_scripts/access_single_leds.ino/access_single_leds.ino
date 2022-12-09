#include <MD_MAX72xx.h>

// Turn on debug statements to the serial output
#define  DEBUG  1

#if  DEBUG
#define PRINT(s, x) { Serial.print(F(s)); Serial.print(x); }
#define PRINTS(x) Serial.print(F(x))
#define PRINTD(x) Serial.println(x, DEC)

#else
#define PRINT(s, x)
#define PRINTS(x)
#define PRINTD(x)

#endif

// Define the number of devices we have in the chain and the hardware interface
// NOTE: These pin numbers will probably not work with your hardware and may
// need to be adapted
#define HARDWARE_TYPE MD_MAX72XX::FC16_HW
#define MAX_DEVICES	16

#define CLK_PIN   13  // or SCK
#define DATA_PIN  11  // or MOSI
#define CS_PIN    10  // or SS

// SPI hardware interface
MD_MAX72XX mx = MD_MAX72XX(HARDWARE_TYPE, CS_PIN, MAX_DEVICES);
// Specific SPI hardware interface
//MD_MAX72XX mx = MD_MAX72XX(HARDWARE_TYPE, SPI1, CS_PIN, MAX_DEVICES);
// Arbitrary pins
//MD_MAX72XX mx = MD_MAX72XX(HARDWARE_TYPE, DATA_PIN, CLK_PIN, CS_PIN, MAX_DEVICES);

// We always wait a bit between updates of the display
#define  DELAYTIME  100  // in milliseconds



void bounce()
// Animation of a bouncing ball
{
  const int minC = 0;
  const int maxC = mx.getColumnCount()-1;
  const int minR = 0;
  const int maxR = ROW_SIZE-1;

  int  nCounter = 0;

  int  r = 0, c = 32;
  int8_t dR = 1, dC = 1;	// delta row and column

  PRINTS("\nBouncing ball");
  // Serial.print(mx.getColumnCount());
  Serial.print(ROW_SIZE);
  mx.clear();

  while (nCounter++ < 1000)
  {
    mx.setPoint(r, c, false);
    // r += dR;
    // c += dC;
    // Serial.print(c);
    mx.setPoint(r, c, true);
    delay(DELAYTIME/2);

    if ((r == minR) || (r == maxR))
      dR = -dR;
    if ((c == minC) || (c == maxC))
      dC = -dC;
  }
}


void setup()
{
  mx.begin();

#if  DEBUG
  Serial.begin(57600);
#endif
  PRINTS("\n[MD_MAX72XX Test & Demo]");
//  scrollText("MD_MAX72xx Test  ");
}

void loop()
{
#if 1
  bounce();
#endif
}