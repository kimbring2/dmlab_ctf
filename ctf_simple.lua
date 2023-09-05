local factory = require 'factories.ctf_simple_factory'

return factory.createLevelApi{
	mapName = 'ctf_simple',
    episodeLengthSeconds = 12,
    botCount = 1,
    level = 1
}