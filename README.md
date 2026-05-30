修改自https://github.com/radiantwf/pokemon-controller-gui
固件在resources/firmware内，在电脑上安装CH341SER.exe。长按BOOT/SELECT键将rp2040与电脑连接，将rp-rust-switch-joystick.uf2刷入rp2040。  
使用时ch340接在电脑上，用杜邦线将ch340的gnd接rp2040的gnd，txd接rp2040的gp1，rxd接rp2040的gp0。rp2040接在switch上进行模拟手柄操作。  
运行脚本时请将模拟手柄排在手柄序列的首位。  
如使用type-c线，使用中如果无反应请检查线是否可以传输数据。  

- 目前仅对剑盾和ZA启用了飞书和meow通知，并设置了obs后台缓存。如果需要请自行修改config.example.json内容并删除文件名中的".example"
- 对ZA中甜甜圈配方设定了默认风味

