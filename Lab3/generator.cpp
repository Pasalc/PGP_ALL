
#include <ios>
#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <math.h>
#include <assert.h>

unsigned long int Endian_DWord_Conversion(uint32_t dword)
{
   return ((dword>>24)&0x000000FF) | ((dword>>8)&0x0000FF00) | ((dword<<8)&0x00FF0000) | ((dword<<24)&0xFF000000);
}

void init_image(uint32_t* i){
    i[0]    =2;
    i[1]    =5;
    i[2]    =3;
    i[3]    =0x280a1900;
    i[4]    =0x32463c00;
    i[5]    =0x5a641900;
}

int main()
{
    std::string str;
    FILE *out;
    out=fopen("cor2.txt", "wb");
    uint32_t *h_image=(uint32_t*)malloc(sizeof(uint32_t)*11);
    init_image(h_image);
    fwrite(h_image,4,3,out);
    
    free(h_image);
    fclose(out);
    return 0;
}
