local factory = require 'factories.ctf_middle_factory'

return factory.createLevelApi{
	mapName = 'ctf_middle',
    episodeLengthSeconds = 1,
    botCount = 1
}